console.log("Content script loaded!", location.href, location.hostname);

// Track our URL changes
let lastUrl = location.href;
let processingUrl = false;
let debounceTimer = null;

// Log initial URL check
console.log("Initial URL check:", {
    href: location.href,
    hostname: location.hostname,
    isInstagram: location.hostname.includes('instagram.com'),
    isReddit: location.hostname.includes('reddit.com'),
    isTwitter: location.hostname.includes('twitter.com') || location.hostname.includes('x.com')
});

// Function to handle URL changes with debouncing
function checkUrlChange() {
    const currentUrl = location.href;
    if (currentUrl !== lastUrl && !processingUrl) {
        lastUrl = currentUrl;
        processingUrl = true;
        
        // Clear any pending debounce
        if (debounceTimer) {
            clearTimeout(debounceTimer);
        }
        
        // Debounce to avoid rapid-fire updates during navigation
        debounceTimer = setTimeout(() => {
            handlePageChange(currentUrl).finally(() => {
                processingUrl = false;
            });
        }, 500); // Wait 500ms after URL change
    }
}

// Observe document changes for SPA navigation
const observer = new MutationObserver(() => {
    checkUrlChange();
});

// Also listen to popstate for browser back/forward
window.addEventListener('popstate', () => {
    checkUrlChange();
});

// Listen to pushstate/replacestate (SPA navigation)
const originalPushState = history.pushState;
const originalReplaceState = history.replaceState;

history.pushState = function(...args) {
    originalPushState.apply(history, args);
    setTimeout(checkUrlChange, 100);
};

history.replaceState = function(...args) {
    originalReplaceState.apply(history, args);
    setTimeout(checkUrlChange, 100);
};

// Now use it to receive notifications
// We want to get the new HTML whenever we enter a new post
observer.observe(document, {
    subtree: true,
    childList: true
});

// Run once on initial load with multiple attempts for slow-loading pages
const initialUrl = location.href;
console.log("Setting up initial load handler for:", initialUrl);

// Try multiple times with increasing delays for slow-loading SPAs like Instagram
setTimeout(() => {
    console.log("Initial load attempt 1 (1s delay)");
    handlePageChange(initialUrl);
}, 1000);

setTimeout(() => {
    if (location.href === initialUrl) {
        console.log("Initial load attempt 2 (3s delay)");
        handlePageChange(initialUrl);
    }
}, 3000);

setTimeout(() => {
    if (location.href === initialUrl) {
        console.log("Initial load attempt 3 (5s delay)");
        handlePageChange(initialUrl);
    }
}, 5000);

// Build full post text for claim detection (same as getClauses input)
function getPostText(postContent) {
    if (!postContent) return "";
    if (postContent.platform === "reddit") {
        return [postContent.title, postContent.body].filter(Boolean).join("\n\n");
    }
    if (postContent.platform === "instagram") return postContent.caption || "";
    if (postContent.platform === "twitter") return postContent.text || "";
    return "";
}

// Our page handler
async function handlePageChange(url) {
    try {
        console.log("Handling page change for URL:", url);
        const postContent = await getPostContent(url);

        if (!postContent) {
            console.warn("No post content extracted for URL:", url);
            return;
        }

        chrome.runtime.sendMessage({
            type: "post-content-updated",
            postContent: postContent,
        }, (response) => {
            if (chrome.runtime.lastError) {
                console.error("Error sending message:", chrome.runtime.lastError);
            } else {
                console.log("Post extracted and sent");
            }
        });
    } catch(error) {
        if (error.message && !error.message.includes("not ready yet") && !error.message.includes("not supported")) {
            console.warn("Error extracting post:", error.message, "URL:", url);
        } else {
            console.debug("Post not ready yet for URL:", url);
        }
    }
}

// Return clauses for the current post. Popup runs the claim model in the browser.
async function runClaimDetection() {
    const url = location.href;
    let postContent = null;
    try {
        postContent = await getPostContent(url);
    } catch (e) {
        return { clauses: [], error: "No post content" };
    }
    if (!postContent) return { clauses: [], error: "No post content" };

    const fullText = getPostText(postContent);
    const clauses = typeof getClauses === "function" ? getClauses(fullText) : [];
    return { clauses };
}

chrome.runtime.onMessage.addListener((message, _sender, sendResponse) => {
    if (message.type === "run-claim-detection") {
        runClaimDetection().then(sendResponse);
        return true;
    }
});

// Helper function to wait for elements to load
function waitForElement(selector, timeout = 5000) {
    return new Promise((resolve, reject) => {
        // Check immediately first
        const el = document.querySelector(selector);
        if (el) {
            resolve(el);
            return;
        }

        const start = Date.now();
        const interval = setInterval(() => {
            const el = document.querySelector(selector);
            if (el) {
                clearInterval(interval);
                resolve(el);
            }
            else if (Date.now() - start > timeout) {
                clearInterval(interval);
                reject(new Error("Timed out waiting for element: " + selector));
            }
        }, 100);
    });
}

// --------------------- Functions to get post content for each site
//  Post getters         These can change frequently, so they are separated and modular
// --------------------- We use Optional Chaining here to avoid any ugly errors

// Branch off into the right social media platform
async function getPostContent(url) {
    // Normalize URL for matching
    const normalizedUrl = url.toLowerCase();
    console.log("getPostContent called with URL:", url, "normalized:", normalizedUrl);
    
    if (normalizedUrl.includes("reddit.com") && normalizedUrl.includes("/comments/")) {
        console.log("Detected Reddit post");
        return await getPostReddit(url);
    } else if (normalizedUrl.includes("instagram.com")) {
        // Instagram can be /p/ or /reel/ or just on instagram.com (SPA)
        console.log("Detected Instagram page");
        return await getPostInstagram();
    } else if ((normalizedUrl.includes("twitter.com") || normalizedUrl.includes("x.com")) && normalizedUrl.includes("/status/")) {
        console.log("Detected Twitter/X post");
        return await getPostTwitter();
    }
    
    console.warn("URL not matched to any platform:", url);
    throw new Error("This site is not supported: " + url);
}

// Reddit HTML structure: Post title is in <h1> on the given page
// Many surrounding (Tailwind CSS?) tags around it, but they are not necessary
async function getPostReddit(url) {
    if (url.includes("old.reddit.com")) {
        throw new Error("Old Reddit is not supported by Roundabout.");
    }
    await waitForElement('h1');

    // Using this structure so we only go through if post content exists
    return {
        platform: "reddit", // Will be used in App.jsx switch statement
        title: document.querySelector('h1')?.innerText ?? "",
        body: document.querySelector('div[slot="text-body"]')?.innerText ?? ""
    }
}

// Instagram HTML structure: Instagram captions can be in various places
// Try multiple selectors as Instagram's structure changes frequently
async function getPostInstagram() {
    console.log("getPostInstagram: Starting extraction");
    
    // Wait for the post to load - Instagram uses article tags
    try {
        await waitForElement('article', 10000);
        console.log("getPostInstagram: Found article element");
    } catch (e) {
        console.warn("getPostInstagram: Article element not found, trying alternative approach");
        // Try waiting a bit more for Instagram's slow loading
        await new Promise(resolve => setTimeout(resolve, 2000));
    }
    
    // Try multiple selectors for Instagram caption
    // Instagram often uses span elements with specific attributes
    const captionSelectors = [
        'article h1',
        'article span[dir="auto"]',
        'article header + div span[dir="auto"]',
        'h1',
        '[data-testid="post-caption"]',
        'article header ~ div span',
        'span[dir="auto"]',
        'article div span',
        'h1[dir="auto"]',
        'article h1 span'
    ];
    
    console.log("getPostInstagram: Trying selectors...");
    let caption = "";
    let foundSelector = null;
    
    for (const selector of captionSelectors) {
        const elements = document.querySelectorAll(selector);
        console.log(`getPostInstagram: Selector "${selector}" found ${elements.length} elements`);
        
        for (const element of elements) {
            const text = element.innerText?.trim() || "";
            if (text.length > 10) {
                // Check if this looks like a caption (not username, not navigation text)
                // Captions are usually longer and don't contain common UI text
                if (!text.includes('Follow') && !text.includes('Message') && !text.includes('More')) {
                    caption = text;
                    foundSelector = selector;
                    console.log(`getPostInstagram: Found caption with selector "${selector}":`, caption.substring(0, 50) + "...");
                    break;
                }
            }
        }
        
        if (caption) break;
    }
    
    // If still no caption, try getting all text from article and finding the longest span
    if (!caption || caption.length === 0) {
        console.log("getPostInstagram: Trying fallback - searching all article text");
        const article = document.querySelector('article');
        if (article) {
            const allSpans = article.querySelectorAll('span');
            let longestText = "";
            for (const span of allSpans) {
                const text = span.innerText?.trim() || "";
                if (text.length > longestText.length && text.length > 20) {
                    // Skip common UI elements
                    if (!text.match(/^\d+[KMB]?$/)) { // Skip numbers like "1.2K"
                        longestText = text;
                    }
                }
            }
            if (longestText.length > 10) {
                caption = longestText;
                foundSelector = "fallback-all-spans";
                console.log("getPostInstagram: Found caption with fallback method");
            }
        }
    }
    
    if (!caption || caption.length === 0) {
        console.error("getPostInstagram: Caption not found. Available elements:", {
            articles: document.querySelectorAll('article').length,
            h1s: document.querySelectorAll('h1').length,
            spans: document.querySelectorAll('span').length
        });
        throw new Error("Instagram caption not found");
    }

    console.log("getPostInstagram: Successfully extracted caption:", caption.substring(0, 100));
    return {
        platform: "instagram",
        caption: caption
    }
}

// Twitter HTML structure: Likewise, for Twitter, we just need the words in the post.
// here we can find the words nested inside the tweet article.
async function getPostTwitter() {
    await waitForElement('article[data-testid="tweet"]');

    return {
        platform: "twitter",
        text: document.querySelector('[data-testid="tweetText"]')?.innerText ?? ""
    }
}
