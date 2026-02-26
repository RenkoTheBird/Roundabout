import { searchBrave } from "../search/search.js";

// Injected at build time by Vite define (no import.meta so service worker works as classic script)
function getBraveApiKey() {
  const key = typeof __BRAVE_API_KEY__ !== "undefined" ? __BRAVE_API_KEY__ : "";
  return key && typeof key === "string" ? key : null;
}

// Handle messages from content script and popup
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Handle post content updates from content script
  if (message.type === "post-content-updated") {
    const tabId = sender.tab?.id;
    if (tabId) {
      chrome.storage.local.set({
        [`post_${tabId}`]: message.postContent,
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

  // Popup requests clauses for current tab; popup runs the claim model in the browser
  if (message.type === "get-claims") {
    const tabId = message.tabId;
    chrome.tabs.sendMessage(tabId, { type: "run-claim-detection" }, (response) => {
      if (chrome.runtime.lastError) {
        sendResponse({ clauses: [], error: chrome.runtime.lastError.message });
      } else {
        sendResponse(response ?? { clauses: [] });
      }
    });
    return true;
  }

  // Popup requests Brave search for a claim query
  if (message.type === "search-claim") {
    const query = message.query;
    const apiKey = getBraveApiKey();
    if (!apiKey) {
      sendResponse({ error: "API key not configured" });
      return;
    }
    searchBrave(query, apiKey)
      .then((out) => {
        if (out.error) {
          sendResponse({ error: out.error });
        } else {
          sendResponse({ results: out.results ?? [] });
        }
      })
      .catch((err) => {
        sendResponse({ error: err?.message || "Search failed" });
      });
    return true;
  }
});
