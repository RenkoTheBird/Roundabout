import { useEffect, useState } from "react";
import './App.css';

function App() {
  const [post, setPost] = useState(null);

  useEffect(() => {
    let currentTabId = null;
    let storageListener = null;

    // Get current tab and request post content
    chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
      if (!tab?.id) return;

      currentTabId = tab.id;

      // Request current post content
      chrome.runtime.sendMessage(
        { type: "get-current-post", tabId: currentTabId },
        (response) => {
          if (response) {
            setPost(response);
          }
        }
      );

      // Listen for storage changes (when content script updates post)
      const storageKey = `post_${currentTabId}`;
      storageListener = (changes, areaName) => {
        if (areaName === "local" && changes[storageKey]) {
          setPost(changes[storageKey].newValue);
        }
      };

      chrome.storage.onChanged.addListener(storageListener);
    });

    // Cleanup function
    return () => {
      if (storageListener) {
        chrome.storage.onChanged.removeListener(storageListener);
      }
    };
  }, []);

  if (!post) {
    return <div className="postFont">No post detected</div>;
  }

  return (
    <>
      <h1>Roundabout</h1>
      <h1>The current post is:</h1>
      <div className="panel">
        <PlatformPost post={post} />
      </div>
    </>
  );
}

function PlatformPost({ post }) {
  switch (post.platform) {
    case "reddit":
      return (
        <>
          <h3 className="postFont">{post.title}</h3>
          <p className="postFont">{post.body}</p>
        </>
      );
    case "instagram":
      return <p className="postFont">{post.caption}</p>;
    case "twitter":
      return <p className="postFont">{post.text}</p>;
    default:
      return <p className="postFont">Unsupported post</p>;
  }
}

export default App;

