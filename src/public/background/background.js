import { searchBrave } from "../search/search.js";

// Injected at build time by Vite define (no import.meta so service worker works as classic script)
function getBraveApiKey() {
  const key = typeof __BRAVE_API_KEY__ !== "undefined" ? __BRAVE_API_KEY__ : "";
  return key && typeof key === "string" ? key : null;
}

const CONTENT_SCRIPT_FILES = ["search/claim-detection.js", "content/content_script.js"];

function shouldInjectIntoTabUrl(urlStr) {
  // Only inject into pages where your existing content script would successfully extract a post.
  try {
    const url = new URL(urlStr);
    if (url.protocol !== "https:") return false;
    const hostname = url.hostname.toLowerCase();
    const path = url.pathname;

    // Reddit posts look like: /comments/<thread_id>/...
    if (hostname === "reddit.com" || hostname === "www.reddit.com") {
      return path.startsWith("/comments/");
    }

    // Instagram posts look like: /p/<id>/..., /reel/<id>/..., /tv/<id>/...
    if (hostname === "instagram.com" || hostname === "www.instagram.com") {
      return path.startsWith("/p/") || path.startsWith("/reel/") || path.startsWith("/tv/");
    }

    // Twitter/X posts look like: /<user>/status/<tweet_id>/...
    if (
      hostname === "x.com" ||
      hostname === "www.x.com" ||
      hostname === "twitter.com" ||
      hostname === "www.twitter.com"
    ) {
      return path.includes("/status/");
    }

    return false;
  } catch {
    return false;
  }
}

function queryTabs() {
  return new Promise((resolve) => {
    chrome.tabs.query({}, (tabs) => resolve(tabs ?? []));
  });
}

function isRoundaboutContentScriptLoaded(tabId) {
  return new Promise((resolve) => {
    chrome.scripting.executeScript(
      {
        target: { tabId, allFrames: false },
        func: () => Boolean(globalThis.__ROUNDABOUT_CONTENT_SCRIPT_LOADED__),
      },
      (results) => {
        if (chrome.runtime.lastError) {
          // If we cannot introspect the page, don't block injection.
          resolve(false);
          return;
        }
        resolve(Boolean(results?.[0]?.result));
      }
    );
  });
}

function injectRoundaboutContentScripts(tabId) {
  return new Promise((resolve, reject) => {
    chrome.scripting.executeScript(
      {
        target: { tabId, allFrames: false },
        files: CONTENT_SCRIPT_FILES,
      },
      () => {
        if (chrome.runtime.lastError) {
          reject(chrome.runtime.lastError);
          return;
        }
        resolve();
      }
    );
  });
}

async function ensureInjectedForTab(tab) {
  const tabId = tab?.id;
  const tabUrl = tab?.url;
  if (!tabId || !tabUrl) return;
  if (!shouldInjectIntoTabUrl(tabUrl)) return;

  try {
    const alreadyInjected = await isRoundaboutContentScriptLoaded(tabId);
    if (alreadyInjected) return;
    await injectRoundaboutContentScripts(tabId);
  } catch (err) {
    // Injection can fail if the URL navigates during injection; don't break the whole loop.
    console.warn("Roundabout: failed to inject content scripts", { tabId, tabUrl, err });
  }
}

async function injectIntoExistingTabs() {
  const tabs = await queryTabs();
  for (const tab of tabs) {
    // Keep this sequential to reduce service-worker lifetime pressure.
    await ensureInjectedForTab(tab);
  }
}

let injectionRunPromise = null;
async function injectIntoExistingTabsOnce() {
  if (injectionRunPromise) return injectionRunPromise;
  injectionRunPromise = injectIntoExistingTabs();
  try {
    await injectionRunPromise;
  } finally {
    injectionRunPromise = null;
  }
}

chrome.runtime.onInstalled.addListener(async () => {
  await injectIntoExistingTabsOnce();
});

chrome.runtime.onStartup.addListener(async () => {
  await injectIntoExistingTabsOnce();
});

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
