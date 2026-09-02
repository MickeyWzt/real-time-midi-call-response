#include <AudioUnitSDK/AUMIDIEffectBase.h>
#include <AudioUnitSDK/AUPlugInDispatch.h>

#include <CoreMIDI/CoreMIDI.h>
#include <AudioToolbox/AudioUnitProperties.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace {

constexpr UInt32 kEnabled = 0;
constexpr UInt32 kResponseSemitones = 1;
constexpr UInt32 kResponseDensity = 2;
constexpr UInt32 kPadRhythm = 3;
constexpr UInt32 kRhythmRecord = 4;
constexpr UInt32 kRhythmLoop = 5;
constexpr UInt32 kRhythmClear = 6;
constexpr UInt32 kDrumChannel = 7;
constexpr UInt32 kParameterCount = 8;
constexpr UInt32 kQueueCapacity = 512;
constexpr UInt32 kRhythmCapacity = 256;
constexpr UInt32 kPacketBufferSize = 4096;

struct QueuedEvent {
	UInt32 status = 0;
	UInt32 data1 = 0;
	UInt32 data2 = 0;
	// Frames from the beginning of the current host render block. Future
	// events stay queued and are reduced once per render block.
	UInt32 framesUntil = 0;
};

struct RhythmEvent {
	UInt8 note = 0;
	UInt8 velocity = 0;
	UInt32 onFrame = 0;
	UInt32 offFrame = 0;
};

class MFPNativeMIDIProcessor final : public ausdk::AUMIDIEffectBase {
public:
	explicit MFPNativeMIDIProcessor(AudioComponentInstance instance)
		: ausdk::AUMIDIEffectBase(instance, false)
	{
		Globals()->SetParameter(kEnabled, 1.0f);
		Globals()->SetParameter(kResponseSemitones, 7.0f);
		Globals()->SetParameter(kResponseDensity, 0.70f);
		Globals()->SetParameter(kPadRhythm, 0.0f);
		Globals()->SetParameter(kRhythmRecord, 0.0f);
		Globals()->SetParameter(kRhythmLoop, 0.0f);
		Globals()->SetParameter(kRhythmClear, 0.0f);
		Globals()->SetParameter(kDrumChannel, 10.0f);
		mActiveRhythm.fill(-1);
	}

	bool StreamFormatWritable(AudioUnitScope, AudioUnitElement) override { return true; }

	OSStatus GetPropertyInfo(AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement element,
		UInt32& outDataSize, bool& outWritable) override
	{
		if (id == kAudioUnitProperty_CocoaUI) {
			if (scope != kAudioUnitScope_Global || element != 0) return kAudioUnitErr_InvalidScope;
			outDataSize = sizeof(AudioUnitCocoaViewInfo);
			outWritable = false;
			return noErr;
		}
		if (id == kAudioUnitProperty_MIDIOutputCallbackInfo || id == kAudioUnitProperty_MIDIOutputCallback) {
			if (scope != kAudioUnitScope_Global || element != 0) return kAudioUnitErr_InvalidScope;
			outDataSize = id == kAudioUnitProperty_MIDIOutputCallbackInfo
				? sizeof(CFArrayRef) : sizeof(AUMIDIOutputCallbackStruct);
			outWritable = id == kAudioUnitProperty_MIDIOutputCallback;
			return noErr;
		}
		return ausdk::AUMIDIEffectBase::GetPropertyInfo(id, scope, element, outDataSize, outWritable);
	}

	OSStatus GetProperty(AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement element,
		void* outData) override
	{
		if (id == kAudioUnitProperty_CocoaUI) {
			if (scope != kAudioUnitScope_Global || element != 0) return kAudioUnitErr_InvalidScope;
			CFBundleRef componentBundle = CFBundleGetBundleWithIdentifier(
				CFSTR("com.mfp.realtime-midi-call-response.midi-fx"));
			if (componentBundle == nullptr) return kAudioUnitErr_InvalidProperty;
			CFURLRef viewBundleURL = CFBundleCopyResourceURL(
				componentBundle, CFSTR("MFPPluginView"), CFSTR("bundle"), nullptr);
			if (viewBundleURL == nullptr) return kAudioUnitErr_InvalidProperty;
			auto* info = static_cast<AudioUnitCocoaViewInfo*>(outData);
			info->mCocoaAUViewBundleLocation = viewBundleURL;
			info->mCocoaAUViewClass[0] = CFSTR("MFPPluginViewFactory");
			CFRetain(info->mCocoaAUViewClass[0]);
			return noErr;
		}
		if (id == kAudioUnitProperty_MIDIOutputCallbackInfo) {
			if (scope != kAudioUnitScope_Global || element != 0) return kAudioUnitErr_InvalidScope;
			CFStringRef name = CFStringCreateWithCString(nullptr, "MFP AI Response", kCFStringEncodingUTF8);
			CFArrayRef array = CFArrayCreate(nullptr, reinterpret_cast<const void**>(&name), 1,
				&kCFTypeArrayCallBacks);
			CFRelease(name);
			*static_cast<CFArrayRef*>(outData) = array;
			return noErr;
		}
		return ausdk::AUMIDIEffectBase::GetProperty(id, scope, element, outData);
	}

	OSStatus SetProperty(AudioUnitPropertyID id, AudioUnitScope scope, AudioUnitElement element,
		const void* data, UInt32 size) override
	{
		if (id == kAudioUnitProperty_MIDIOutputCallback) {
			if (scope != kAudioUnitScope_Global || element != 0 || size < sizeof(AUMIDIOutputCallbackStruct)) {
				return kAudioUnitErr_InvalidPropertyValue;
			}
			mOutput = *static_cast<const AUMIDIOutputCallbackStruct*>(data);
			return noErr;
		}
		return ausdk::AUMIDIEffectBase::SetProperty(id, scope, element, data, size);
	}

	OSStatus GetParameterList(AudioUnitScope scope, AudioUnitParameterID* ids, UInt32& count) override
	{
		if (scope != kAudioUnitScope_Global) return kAudioUnitErr_InvalidScope;
		if (ids == nullptr) { count = kParameterCount; return noErr; }
		if (count < kParameterCount) return kAudio_ParamError;
		for (UInt32 i = 0; i < kParameterCount; ++i) ids[i] = i;
		count = kParameterCount;
		return noErr;
	}

	OSStatus GetParameterInfo(AudioUnitScope scope, AudioUnitParameterID id,
		AudioUnitParameterInfo& info) override
	{
		if (scope != kAudioUnitScope_Global || id >= kParameterCount) return kAudioUnitErr_InvalidParameter;
		info = {};
		info.flags = kAudioUnitParameterFlag_IsWritable | kAudioUnitParameterFlag_IsReadable;
		switch (id) {
		case kEnabled:
			FillInParameterName(info, CFSTR("MFP Enabled"), false);
			info.unit = kAudioUnitParameterUnit_Boolean; info.minValue = 0.0f; info.maxValue = 1.0f; info.defaultValue = 1.0f; break;
		case kResponseSemitones:
			FillInParameterName(info, CFSTR("Response Semitones"), false);
			info.unit = kAudioUnitParameterUnit_Indexed; info.minValue = -12.0f; info.maxValue = 12.0f; info.defaultValue = 7.0f; break;
		case kResponseDensity:
			FillInParameterName(info, CFSTR("Response Density"), false);
			info.unit = kAudioUnitParameterUnit_Generic; info.minValue = 0.1f; info.maxValue = 1.0f; info.defaultValue = 0.7f; break;
		case kPadRhythm:
			FillInParameterName(info, CFSTR("Pad Rhythm"), false);
			info.unit = kAudioUnitParameterUnit_Boolean; info.minValue = 0.0f; info.maxValue = 1.0f; info.defaultValue = 0.0f; break;
		case kRhythmRecord:
			FillInParameterName(info, CFSTR("Rhythm Record"), false);
			info.unit = kAudioUnitParameterUnit_Boolean; info.minValue = 0.0f; info.maxValue = 1.0f; info.defaultValue = 0.0f; break;
		case kRhythmLoop:
			FillInParameterName(info, CFSTR("Rhythm Loop"), false);
			info.unit = kAudioUnitParameterUnit_Boolean; info.minValue = 0.0f; info.maxValue = 1.0f; info.defaultValue = 0.0f; break;
		case kRhythmClear:
			FillInParameterName(info, CFSTR("Rhythm Clear"), false);
			info.unit = kAudioUnitParameterUnit_Boolean; info.minValue = 0.0f; info.maxValue = 1.0f; info.defaultValue = 0.0f; break;
		case kDrumChannel:
			FillInParameterName(info, CFSTR("Drum Channel"), false);
			info.unit = kAudioUnitParameterUnit_Indexed; info.minValue = 1.0f; info.maxValue = 16.0f; info.defaultValue = 10.0f; break;
		default: return kAudioUnitErr_InvalidParameter;
		}
		return noErr;
	}

	OSStatus Render(AudioUnitRenderActionFlags& flags, const AudioTimeStamp& time, UInt32 frames) override
	{
		(void)flags;
		if (GetParameter(kRhythmClear) >= 0.5f) {
			ClearRhythm();
			Globals()->SetParameter(kRhythmClear, 0.0f);
		}
		UpdateRhythmState();
		if (GetParameter(kRhythmRecord) < 0.5f) ScheduleRhythmLoop(frames);
		if (mOutput.midiOutputCallback == nullptr || mQueuedCount == 0) {
			mCurrentFrame += frames;
			return noErr;
		}
		std::array<UInt8, kPacketBufferSize> storage{};
		auto* packetList = reinterpret_cast<MIDIPacketList*>(storage.data());
		MIDIPacket* packet = MIDIPacketListInit(packetList);
		UInt32 emitted = 0;
		std::array<QueuedEvent, kQueueCapacity> remaining{};
		UInt32 remainingCount = 0;
		for (UInt32 i = 0; i < mQueuedCount; ++i) {
			QueuedEvent event = mQueue[i];
			if (event.framesUntil >= frames || emitted >= 96) {
				if (event.framesUntil >= frames) event.framesUntil -= frames;
				if (remainingCount < kQueueCapacity) remaining[remainingCount++] = event;
				continue;
			}
			const UInt8 bytes[] = {static_cast<UInt8>(event.status), static_cast<UInt8>(event.data1), static_cast<UInt8>(event.data2)};
			packet = MIDIPacketListAdd(packetList, kPacketBufferSize, packet, event.framesUntil, 3, bytes);
			if (packet == nullptr) {
				if (remainingCount < kQueueCapacity) remaining[remainingCount++] = event;
				continue;
			}
			++emitted;
		}
		mQueue = remaining;
		mQueuedCount = remainingCount;
		mCurrentFrame += frames;
		return emitted == 0 ? noErr : mOutput.midiOutputCallback(mOutput.userData, &time, 0, packetList);
	}

protected:
	OSStatus HandleNoteOn(UInt8 channel, UInt8 note, UInt8 velocity, UInt32 offset) override
	{
		// The model and rhythm guide run in the companion service. The AU is a
		// control surface and clean MIDI bridge, so it must never add a second
		// synthetic response on top of the service output.
		Enqueue(0x90U | channel, note, velocity, offset);
		return noErr;
	}

	OSStatus HandleNoteOff(UInt8 channel, UInt8 note, UInt8 velocity, UInt32 offset) override
	{
		// Always forward source Note Off so a bypass or service state change can
		// never leave a note hanging in Logic.
		Enqueue(0x80U | channel, note, velocity, offset);
		return noErr;
	}

	OSStatus HandleControlChange(UInt8 channel, UInt8 controller, UInt8 value, UInt32 offset) override
	{
		Enqueue(0xB0U | channel, controller, value, offset);
		return noErr;
	}

	OSStatus HandleAllNotesOff(UInt8 channel) override
	{
		ClearQueuedNotes(channel);
		Enqueue(0xB0U | channel, 123U, 0U, 0U);
		return noErr;
	}

	OSStatus HandleAllSoundOff(UInt8 channel) override
	{
		ClearQueuedNotes(channel);
		Enqueue(0xB0U | channel, 120U, 0U, 0U);
		Enqueue(0xB0U | channel, 123U, 0U, 0U);
		return noErr;
	}

private:
	void Enqueue(UInt32 status, UInt32 data1, UInt32 data2, UInt32 offset) AUSDK_RTSAFE
	{
		if (mQueuedCount >= kQueueCapacity) return;
		UInt32 position = mQueuedCount;
		while (position > 0 && mQueue[position - 1].framesUntil > offset) {
			mQueue[position] = mQueue[position - 1];
			--position;
		}
		mQueue[position] = {status, data1, data2, offset};
		++mQueuedCount;
	}

	void ClearQueuedNotes(UInt8 channel) AUSDK_RTSAFE
	{
		const UInt32 wanted = 0x90U | channel;
		const UInt32 wantedOff = 0x80U | channel;
		std::array<QueuedEvent, kQueueCapacity> remaining{};
		UInt32 remainingCount = 0;
		for (UInt32 i = 0; i < mQueuedCount; ++i) {
			const QueuedEvent event = mQueue[i];
			if (event.status != wanted && event.status != wantedOff && remainingCount < kQueueCapacity) remaining[remainingCount++] = event;
		}
		mQueue = remaining;
		mQueuedCount = remainingCount;
	}

	UInt8 DrumChannel() AUSDK_RTSAFE
	{
		const auto value = std::clamp(GetParameter(kDrumChannel), 1.0f, 16.0f);
		return static_cast<UInt8>(std::round(value) - 1.0f);
	}

	UInt32 DefaultNoteLength() AUSDK_RTSAFE
	{
		return std::max<UInt32>(1U, static_cast<UInt32>(GetSampleRate() * 0.12));
	}

	void RecordRhythmNoteOn(UInt8 note, UInt8 velocity, UInt32 offset) AUSDK_RTSAFE
	{
		if (GetParameter(kRhythmRecord) < 0.5f || mRhythmCount >= kRhythmCapacity) return;
		const UInt64 absolute = mCurrentFrame + offset;
		if (!mRhythmRecording) {
			mRhythmRecording = true;
			mRhythmOriginFrame = absolute;
		}
		const UInt32 relative = static_cast<UInt32>(absolute - mRhythmOriginFrame);
		mRhythm[mRhythmCount] = {note, velocity, relative, relative + DefaultNoteLength()};
		mActiveRhythm[note - 36] = static_cast<std::int32_t>(mRhythmCount);
		++mRhythmCount;
		mRhythmLengthFrames = std::max(mRhythmLengthFrames, relative + DefaultNoteLength());
	}

	void RecordRhythmNoteOff(UInt8 note, UInt32 offset) AUSDK_RTSAFE
	{
		if (!mRhythmRecording || note < 36 || note > 51) return;
		const std::int32_t index = mActiveRhythm[note - 36];
		if (index < 0 || static_cast<UInt32>(index) >= mRhythmCount) return;
		const UInt64 absolute = mCurrentFrame + offset;
		const UInt32 relative = static_cast<UInt32>(absolute - mRhythmOriginFrame);
		mRhythm[index].offFrame = std::max(mRhythm[index].onFrame + 1U, relative);
		mRhythmLengthFrames = std::max(mRhythmLengthFrames, mRhythm[index].offFrame);
		mActiveRhythm[note - 36] = -1;
	}

	void ClearRhythm() AUSDK_RTSAFE
	{
		mRhythmCount = 0;
		mRhythmLengthFrames = 0;
		mRhythmRecording = false;
		mRhythmLoopActive = false;
		mLoopScheduledThrough = 0;
		mRhythmOriginFrame = mCurrentFrame;
		mActiveRhythm.fill(-1);
	}

	void UpdateRhythmState() AUSDK_RTSAFE
	{
		const bool loop = GetParameter(kRhythmLoop) >= 0.5f;
		if (loop && !mRhythmLoopActive && mRhythmCount > 0) {
			mRhythmOriginFrame = mCurrentFrame;
			mLoopScheduledThrough = mCurrentFrame;
		}
		if (!loop && mRhythmLoopActive) {
			Enqueue(0xB0U | DrumChannel(), 123U, 0U, 0U);
		}
		mRhythmLoopActive = loop;
	}

	void ScheduleRhythmLoop(UInt32 frames) AUSDK_RTSAFE
	{
		if (!mRhythmLoopActive || mRhythmCount == 0 || mRhythmLengthFrames == 0) return;
		const UInt64 horizon = mCurrentFrame + frames * 2ULL + mRhythmLengthFrames;
		while (mLoopScheduledThrough < horizon) {
			for (UInt32 i = 0; i < mRhythmCount; ++i) {
				const auto& hit = mRhythm[i];
				const UInt64 on = mLoopScheduledThrough + hit.onFrame;
				const UInt64 off = mLoopScheduledThrough + hit.offFrame;
				if (on >= mCurrentFrame) {
					Enqueue(0x90U | DrumChannel(), hit.note, hit.velocity, static_cast<UInt32>(on - mCurrentFrame));
					Enqueue(0x80U | DrumChannel(), hit.note, 0U, static_cast<UInt32>(off - mCurrentFrame));
				}
			}
			mLoopScheduledThrough += mRhythmLengthFrames;
		}
	}

	AUMIDIOutputCallbackStruct mOutput{};
	std::array<QueuedEvent, kQueueCapacity> mQueue{};
	UInt32 mQueuedCount = 0;
	std::array<RhythmEvent, kRhythmCapacity> mRhythm{};
	std::array<std::int32_t, 16> mActiveRhythm{};
	UInt32 mRhythmCount = 0;
	UInt32 mRhythmLengthFrames = 0;
	UInt64 mCurrentFrame = 0;
	UInt64 mRhythmOriginFrame = 0;
	UInt64 mLoopScheduledThrough = 0;
	bool mRhythmRecording = false;
	bool mRhythmLoopActive = false;
};

} // namespace

AUSDK_COMPONENT_ENTRY(ausdk::AUMIDIEffectFactory, MFPNativeMIDIProcessor)
