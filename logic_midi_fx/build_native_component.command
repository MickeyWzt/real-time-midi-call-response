#!/bin/zsh
set -euo pipefail

PROJECT_DIR="${0:A:h}"
ROOT_DIR="${PROJECT_DIR:h}"
SDK_DIR="${ROOT_DIR}/.native_sdk"
MAC_SDK="$(xcrun --sdk macosx --show-sdk-path)"
BUILD_DIR="${PROJECT_DIR}/build"
COMPONENT_DIR="${BUILD_DIR}/MFPNativeMIDIProcessor.component"
CONTENTS_DIR="${COMPONENT_DIR}/Contents"
MACOS_DIR="${CONTENTS_DIR}/MacOS"
RESOURCES_DIR="${CONTENTS_DIR}/Resources"
VIEW_BUNDLE_DIR="${RESOURCES_DIR}/MFPPluginView.bundle"
VIEW_CONTENTS_DIR="${VIEW_BUNDLE_DIR}/Contents"
VIEW_MACOS_DIR="${VIEW_CONTENTS_DIR}/MacOS"

mkdir -p "${MACOS_DIR}" "${VIEW_MACOS_DIR}"
rm -f "${MACOS_DIR}/MFPNativeMIDIProcessor"
rm -f "${VIEW_MACOS_DIR}/MFPPluginView"

COMMON=(-arch arm64 -isysroot "${MAC_SDK}" -mmacosx-version-min=11.0 -std=c++23
  -I"${MAC_SDK}/usr/include/c++/v1" -I"${SDK_DIR}/include" -fvisibility=hidden -fPIC -O2
  -D_AU_DEBUG_=0 -DAUSDK_HAVE_UI=1 -DAUSDK_HAVE_MIDI2=0)

SDK_SOURCES=(
  "${SDK_DIR}/src/AudioUnitSDK/AUBase.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUBuffer.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUBufferAllocator.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUEffectBase.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUInputElement.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUMIDIBase.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUMIDIEffectBase.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUOutputElement.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUPlugInDispatch.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/AUScopeElement.cpp"
  "${SDK_DIR}/src/AudioUnitSDK/ComponentBase.cpp"
)

OBJECTS=()
for source in "${SDK_SOURCES[@]}" "${PROJECT_DIR}/MFPNativeMIDIProcessor.cpp"; do
  object="${BUILD_DIR}/$(basename "${source}" .cpp).o"
  clang++ "${COMMON[@]}" -c "${source}" -o "${object}"
  OBJECTS+=("${object}")
done

clang++ -dynamiclib "${COMMON[@]}" "${OBJECTS[@]}" -framework AudioToolbox -framework CoreFoundation \
  -framework CoreMIDI -o "${MACOS_DIR}/MFPNativeMIDIProcessor"

cp "${PROJECT_DIR}/Info.plist" "${CONTENTS_DIR}/Info.plist"
clang++ -arch arm64 -isysroot "${MAC_SDK}" -mmacosx-version-min=11.0 -fobjc-arc -fvisibility=hidden \
  -bundle "${PROJECT_DIR}/MFPPluginView.mm" -framework AudioToolbox -framework Cocoa -framework WebKit \
  -o "${VIEW_MACOS_DIR}/MFPPluginView"
cp "${PROJECT_DIR}/MFPPluginView-Info.plist" "${VIEW_CONTENTS_DIR}/Info.plist"
codesign --force --sign - "${VIEW_BUNDLE_DIR}" >/dev/null
codesign --force --deep --sign - "${COMPONENT_DIR}" >/dev/null

print "Built ${COMPONENT_DIR}"
