// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle post content updates from content script
  if (message.type === "post-content-updated") {
    const tabId = sender.tab?.id;
    if (tabId) {
      // Store in chrome.storage so popup can listen for changes
      chrome.storage.local.set({
        [`post_${tabId}`]: message.postContent
      });
    }
    return;
  }

  // Handle requests from popup for current post
  if (message.type === "get-current-post") {
    const tabId = message.tabId;
    chrome.storage.local.get([`post_${tabId}`], (result) => {
      sendResponse(result[`post_${tabId}`] ?? null);
    });
    return true;
  }
});
