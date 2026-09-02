#import <AudioToolbox/AUCocoaUIView.h>
#import <Cocoa/Cocoa.h>
#import <WebKit/WebKit.h>

static NSString *const MFPControlURL = @"http://127.0.0.1:8000/?embedded=logic";

@interface MFPPluginContainerView : NSView <WKNavigationDelegate>
@property(nonatomic, strong) WKWebView *webView;
@property(nonatomic, strong) NSView *offlinePanel;
@property(nonatomic, strong) NSTextField *panelTitle;
@property(nonatomic, strong) NSTextField *panelDetail;
@property(nonatomic, strong) NSProgressIndicator *loadingIndicator;
@property(nonatomic, strong) NSButton *retryButton;
@end

@implementation MFPPluginContainerView

- (instancetype)initWithFrame:(NSRect)frameRect
{
    self = [super initWithFrame:frameRect];
    if (self == nil) return nil;

    self.wantsLayer = YES;
    self.layer.backgroundColor = [NSColor colorWithWhite:0.055 alpha:1.0].CGColor;

    WKWebViewConfiguration *configuration = [[WKWebViewConfiguration alloc] init];
    self.webView = [[WKWebView alloc] initWithFrame:self.bounds configuration:configuration];
    self.webView.autoresizingMask = NSViewWidthSizable | NSViewHeightSizable;
    self.webView.navigationDelegate = self;
    [self addSubview:self.webView];

    NSView *panel = [[NSView alloc] initWithFrame:NSMakeRect(0, 0, 420, 180)];
    panel.wantsLayer = YES;
    panel.layer.backgroundColor = [NSColor colorWithWhite:0.09 alpha:0.98].CGColor;
    panel.layer.cornerRadius = 6.0;
    panel.translatesAutoresizingMaskIntoConstraints = NO;
    self.offlinePanel = panel;
    [self addSubview:panel];

    NSTextField *title = [NSTextField labelWithString:@"MFP 服务未连接"];
    title.font = [NSFont systemFontOfSize:17 weight:NSFontWeightSemibold];
    title.textColor = NSColor.whiteColor;
    title.alignment = NSTextAlignmentCenter;
    title.translatesAutoresizingMaskIntoConstraints = NO;
    self.panelTitle = title;
    [panel addSubview:title];

    NSTextField *detail = [NSTextField labelWithString:@"请启动 MFP 后台，然后重试连接。"];
    detail.textColor = [NSColor colorWithWhite:0.68 alpha:1.0];
    detail.alignment = NSTextAlignmentCenter;
    detail.translatesAutoresizingMaskIntoConstraints = NO;
    self.panelDetail = detail;
    [panel addSubview:detail];

    NSProgressIndicator *indicator = [[NSProgressIndicator alloc] initWithFrame:NSZeroRect];
    indicator.style = NSProgressIndicatorStyleSpinning;
    indicator.controlSize = NSControlSizeSmall;
    indicator.indeterminate = YES;
    indicator.translatesAutoresizingMaskIntoConstraints = NO;
    self.loadingIndicator = indicator;
    [panel addSubview:indicator];

    NSButton *retry = [NSButton buttonWithTitle:@"重试" target:self action:@selector(loadControlSurface)];
    retry.bezelStyle = NSBezelStyleRounded;
    retry.translatesAutoresizingMaskIntoConstraints = NO;
    self.retryButton = retry;
    [panel addSubview:retry];

    [NSLayoutConstraint activateConstraints:@[
        [panel.centerXAnchor constraintEqualToAnchor:self.centerXAnchor],
        [panel.centerYAnchor constraintEqualToAnchor:self.centerYAnchor],
        [panel.widthAnchor constraintEqualToConstant:420],
        [panel.heightAnchor constraintEqualToConstant:180],
        [title.topAnchor constraintEqualToAnchor:panel.topAnchor constant:26],
        [title.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:20],
        [title.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-20],
        [detail.topAnchor constraintEqualToAnchor:title.bottomAnchor constant:10],
        [detail.leadingAnchor constraintEqualToAnchor:panel.leadingAnchor constant:20],
        [detail.trailingAnchor constraintEqualToAnchor:panel.trailingAnchor constant:-20],
        [indicator.topAnchor constraintEqualToAnchor:detail.bottomAnchor constant:18],
        [indicator.centerXAnchor constraintEqualToAnchor:panel.centerXAnchor],
        [retry.topAnchor constraintEqualToAnchor:detail.bottomAnchor constant:14],
        [retry.centerXAnchor constraintEqualToAnchor:panel.centerXAnchor],
    ]];

    [self loadControlSurface];
    return self;
}

- (void)loadControlSurface
{
    self.panelTitle.stringValue = @"正在连接 MFP";
    self.panelDetail.stringValue = @"正在加载 Logic 内的实时控制界面。";
    self.retryButton.hidden = YES;
    self.loadingIndicator.hidden = NO;
    [self.loadingIndicator startAnimation:nil];
    self.offlinePanel.hidden = NO;
    [self addSubview:self.offlinePanel positioned:NSWindowAbove relativeTo:self.webView];
    NSURL *url = [NSURL URLWithString:MFPControlURL];
    NSURLRequest *request = [NSURLRequest requestWithURL:url
        cachePolicy:NSURLRequestReloadIgnoringLocalCacheData
        timeoutInterval:5.0];
    [self.webView loadRequest:request];
}

- (void)showOfflinePanel
{
    self.panelTitle.stringValue = @"MFP 服务未连接";
    self.panelDetail.stringValue = @"请启动 MFP 后台，然后重试连接。";
    [self.loadingIndicator stopAnimation:nil];
    self.loadingIndicator.hidden = YES;
    self.retryButton.hidden = NO;
    self.offlinePanel.hidden = NO;
    [self addSubview:self.offlinePanel positioned:NSWindowAbove relativeTo:self.webView];
}

- (void)webView:(WKWebView *)webView didFinishNavigation:(WKNavigation *)navigation
{
    (void)webView;
    (void)navigation;
    [self.loadingIndicator stopAnimation:nil];
    self.offlinePanel.hidden = YES;
}

- (void)webView:(WKWebView *)webView didFailNavigation:(WKNavigation *)navigation
    withError:(NSError *)error
{
    (void)webView;
    (void)navigation;
    (void)error;
    [self showOfflinePanel];
}

- (void)webView:(WKWebView *)webView didFailProvisionalNavigation:(WKNavigation *)navigation
    withError:(NSError *)error
{
    (void)webView;
    (void)navigation;
    (void)error;
    [self showOfflinePanel];
}

- (void)webViewWebContentProcessDidTerminate:(WKWebView *)webView
{
    (void)webView;
    [self showOfflinePanel];
}

@end

@interface MFPPluginViewFactory : NSObject <AUCocoaUIBase>
@end

@implementation MFPPluginViewFactory

- (unsigned)interfaceVersion
{
    return 0;
}

- (NSString *)description
{
    return @"MFP Live Studio";
}

- (NSView *)uiViewForAudioUnit:(AudioUnit)audioUnit withSize:(NSSize)preferredSize
{
    (void)audioUnit;
    CGFloat width = preferredSize.width >= 700 ? preferredSize.width : 1180;
    CGFloat height = preferredSize.height >= 500 ? preferredSize.height : 760;
    return [[MFPPluginContainerView alloc] initWithFrame:NSMakeRect(0, 0, width, height)];
}

@end
