import { Button } from "./ui/button"
import { Card } from "./ui/card"
import { Textarea } from "./ui/textarea"
import { ScrollArea } from "./ui/scroll-area"
import { useNavigate, useParams } from "react-router-dom"
import { SendHorizontal } from "lucide-react"
import { useState, useRef, useEffect } from "react"
import { useWebSocket } from "../context/WebSocketContext"
import { ChatNavbar } from "./ChatNavbar"
import MarkdownPreview from "@uiw/react-markdown-preview"
import rehypeExternalLinks from 'rehype-external-links'
import { Toaster } from "./ui/sonner"
import { toast } from "sonner"

interface Message {
  content: string
  role: 'user' | 'assistant'
}

export function Chat() {
  const navigate = useNavigate()
  const { owner, repo } = useParams<{ owner: string; repo: string }>()
  const { isConnected, sendMessage, lastMessage, disconnect, isProcessing, connect } = useWebSocket()
  const [messages, setMessages] = useState<Message[]>([])
  const [inputValue, setInputValue] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [connectionError, setConnectionError] = useState<string>("")
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const connectionAttemptedRef = useRef(false)
  const welcomeMessageShownRef = useRef(false)

  // Add function to transform markdown content
  const transformMarkdown = (content: string) => {
    // Transform relative links like [CONTRIBUTING.md](CONTRIBUTING.md) to full GitHub URLs
    return content.replace(
      /\[([^\]]+)\]\((?!https?:\/\/)([^)]+)\)/g,
      `[$1](https://github.com/${owner}/${repo}/blob/main/$2)`
    )
  }

  // Add cleanup effect
  useEffect(() => {
    // Cleanup function that runs on unmount
    return () => {
      disconnect()
      setMessages([])
      connectionAttemptedRef.current = false
      welcomeMessageShownRef.current = false
      // Clear history state to prevent restoration
      window.history.replaceState({}, document.title)
    }
  }, [])

  // Block back navigation if WebSocket is connected
  useEffect(() => {
    const handleBeforeUnload = () => {
      if (isConnected) {
        disconnect()
      }
    }

    const handlePopState = () => {
      if (isConnected) {
        disconnect()
        setMessages([])
      }
    }

    window.addEventListener('beforeunload', handleBeforeUnload)
    window.addEventListener('popstate', handlePopState)

    return () => {
      window.removeEventListener('beforeunload', handleBeforeUnload)
      window.removeEventListener('popstate', handlePopState)
    }
  }, [isConnected])

  // Initialize WebSocket connection when component mounts
  useEffect(() => {
    const initConnection = async () => {
      if (!owner || !repo || connectionAttemptedRef.current) {
        return
      }

      connectionAttemptedRef.current = true
      try {
        await connect(owner, repo)
        setConnectionError("")
      } catch (err) {
        console.error('Failed to connect:', err)
        setConnectionError("Failed to connect to the repository. Please try again.")
      }
    }

    initConnection()
    return () => {
      disconnect()
      connectionAttemptedRef.current = false
      welcomeMessageShownRef.current = false
    }
  }, [owner, repo])

  const addMessage = (content: string, role: 'user' | 'assistant') => {
    setMessages(prev => [...prev, { content, role }])
  }

  // Focus input on mount and whenever it should be active
  useEffect(() => {
    if (!isLoading && !isProcessing && isConnected) {
      inputRef.current?.focus();
    }
  }, [isLoading, isProcessing, isConnected]);

  // Handle WebSocket messages
  useEffect(() => {
    if (lastMessage === 'repo_processed' && !welcomeMessageShownRef.current) {
      welcomeMessageShownRef.current = true;
      // Use Sonner's success toast for repo processed
      toast.success("Repository processed!", {
        position: "top-center",
        style: {
          background: "#bde851",
          color: "#222",
          fontWeight: 600,
          borderRadius: 12,
          boxShadow: "0 2px 16px 0 rgba(0,0,0,0.08)",
          maxWidth: "400px",
          width: "100%",
          textAlign: "center",
          padding: "0.75rem 1.5rem",
          margin: "0 auto",
        },
        duration: 3200,
      });
      addMessage("Hello! I've analyzed this repository. What would you like to know?", 'assistant');
    } else if (lastMessage && lastMessage !== 'repo_processed') {
      addMessage(lastMessage, 'assistant');
      setIsLoading(false);
    }
  }, [lastMessage]);

  // Scroll to bottom when messages change or loading state changes
  const [showScrollButton, setShowScrollButton] = useState(false);
  // Remove the old scrollToBottom function and its useEffect
  
  // Add new scroll position tracking
  const handleScroll = () => {
    const viewport = document.querySelector('[data-radix-scroll-area-viewport]');
    if (viewport) {
      const { scrollTop, scrollHeight, clientHeight } = viewport;
      const isAtBottom = scrollHeight - scrollTop - clientHeight < 50;
      setShowScrollButton(!isAtBottom);
    }
  };
  
  useEffect(() => {
    const viewport = document.querySelector('[data-radix-scroll-area-viewport]');
    if (viewport) {
      viewport.addEventListener('scroll', handleScroll);
      return () => viewport.removeEventListener('scroll', handleScroll);
    }
  }, []);
  
  const scrollToBottom = () => {
    const viewport = document.querySelector('[data-radix-scroll-area-viewport]');
    if (viewport) {
      viewport.scrollTo({
        top: viewport.scrollHeight,
        behavior: 'smooth'
      });
    }
  };

  useEffect(() => {
    scrollToBottom()
  }, [messages, isLoading])

  // Remove the auto-scroll useEffect
  const handleSend = () => {
    if (!inputValue.trim() || !isConnected || isLoading || isProcessing) return;

    addMessage(inputValue, 'user');
    setIsLoading(true);
    sendMessage(inputValue);
    setInputValue("");
    inputRef.current?.focus();
  };

  if (connectionError || lastMessage?.startsWith('error:') || lastMessage?.startsWith('All API keys')) {
    const isRepoTooLarge = lastMessage === 'error:repo_too_large';
    const isRepoNotFound = lastMessage === 'error:repo_not_found';
    const isRepoPrivate = lastMessage === 'error:repo_private';
    const isKeysExhausted = lastMessage?.startsWith('All API keys');

    return (
      <div className="min-h-screen w-full flex flex-col bg-background">
        <ChatNavbar 
          onNewChat={() => {
            setMessages([])
            disconnect()
            navigate('/')
          }}
        />
        <div className="flex-1 flex items-center justify-center p-4">
          <Card className="max-w-[600px] w-full p-6 sm:p-8">
            <h2 className="text-xl sm:text-2xl font-semibold mb-4">
              {isRepoTooLarge 
                ? "Repository Size Limit Exceeded"
                : isRepoNotFound
                ? "Repository Not Found"
                : isRepoPrivate
                ? "Private Repository"
                : isKeysExhausted
                ? "Service Temporarily Unavailable"
                : "Connection Failed"}
            </h2>
            <p className="text-foreground/80 whitespace-pre-line mb-6">
              {isRepoTooLarge
                ? "Support for larger repositories is coming very soon! 🚀\n\nCurrently this repository exceeds our size limits, but we're actively working on expanding TalkToGitHub's capabilities. In the meantime, you can try:\n\n• Using a smaller repository\n• Starting with the main branch only\n• Check back soon - large repository support is a top priority!"
                : isRepoNotFound
                ? "The repository you're trying to access doesn't seem to exist. This could be because:\n\n• The repository URL is incorrect\n• The repository has been deleted or moved\n• You made a typo in the owner or repository name"
                : isRepoPrivate
                ? "This appears to be a private repository that we can't access. Currently, TalkToGitHub only works with public repositories. You can:\n\n• Try using a public repository instead\n• Make this repository public if you own it\n• Check back later - private repository support is on our roadmap!"
                : isKeysExhausted
                ? "TTG is temporarily down! We are actively working on a fix and it will be up soon! 🛠️\n\nIn the meantime:\n\n• <a href='https://x.com/HarishChan3304' target='_blank' rel='noopener noreferrer' class='text-main hover:underline'>Check X.com for updates</a>\n• Try again in a few minutes\n• Consider starring the project on GitHub to stay updated"
                : "Unable to establish connection to the repository. This could be due to:\n\n• Server connectivity issues\n• Repository access restrictions\n• Temporary service disruption"}
            </p>
            <div className="flex flex-col sm:flex-row gap-3">
              <Button 
                onClick={() => navigate('/')}
                className="flex-1"
                size="lg"
              >
                Try Another Repository
              </Button>
              {!isRepoNotFound && !isRepoPrivate && (
                <Button 
                  variant="neutral"
                  onClick={() => window.location.reload()}
                  className="flex-1"
                  size="lg"
                >
                  Try Again
                </Button>
              )}
            </div>
            <div className="mt-6 pt-6 border-t-2 border-border">
              <p className="text-sm text-foreground/70 text-center">
                {isRepoTooLarge 
                  ? "Have a large repository you'd like to analyze?"
                  : isRepoPrivate
                  ? "Need private repository support?"
                  : isKeysExhausted
                  ? "Want to support TTG?"
                  : "Having trouble connecting?"}
                <a 
                  href={isKeysExhausted 
                    ? "https://github.com/HarishChandran3304/TTG"
                    : "https://github.com/HarishChandran3304/TTG/issues/new"} 
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-main hover:underline ml-1"
                >
                  {isKeysExhausted ? "Star us on GitHub" : "Let us know"}
                </a>
              </p>
            </div>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen w-full flex flex-col bg-background">
      <Toaster richColors={false} />
      <ChatNavbar 
        onNewChat={() => {
          setMessages([])
          disconnect()
          navigate('/')
        }}
      />

      {/* Chat Container */}
      <div className="flex-1 w-full mx-auto max-w-[900px]">
        <ScrollArea className="h-[calc(100vh-12rem)]" onScrollCapture={handleScroll}>
          <div className="py-4 px-4 sm:px-8 space-y-6">
            {/* Repository Info */}
            <div className="flex justify-center mb-8">
              <Card className="inline-flex items-center gap-2 px-3 py-2 bg-secondary-background/50 border-2 border-border shadow-shadow">
                <svg height="16" viewBox="0 0 16 16" version="1.1" width="16" className="text-foreground">
                  <path fill="currentColor" d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"></path>
                </svg>
                <a 
                  href={`https://github.com/${owner}/${repo}`}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-sm font-medium hover:underline text-foreground"
                >
                  {owner}/{repo}
                </a>
              </Card>
            </div>

            {/* Repository Processing Message */}
            {isProcessing && (
              <div className="flex justify-center mb-8">
                <Card className="p-4 rounded-2xl border-2 border-border bg-secondary-background/50 shadow-shadow animate-pulse">
                  <p className="text-[15px] text-foreground">Processing repository... This may take a few seconds.</p>
                </Card>
              </div>
            )}

            {/* Message List */}
            {messages.map((message, index) => (
              message.role === "user" ? (
                // User Message
                <div key={index} className="flex justify-end">
                  <Card className="max-w-[85%] p-3 sm:p-4 rounded-2xl border-2 border-border bg-main shadow-shadow">
                    <p className="text-[14px] sm:text-[15px] text-main-foreground">{message.content}</p>
                  </Card>
                </div>
              ) : (
                // Bot Message
                <div key={index} className="flex">
                  <Card className="max-w-[85%] p-3 sm:p-4 rounded-2xl border-2 border-border bg-secondary-background shadow-shadow">
                    <div className="wmde-markdown-var overflow-hidden">
                      <MarkdownPreview 
                        source={transformMarkdown(message.content)}
                        rehypePlugins={[[rehypeExternalLinks, { target: '_blank', rel: 'noopener noreferrer' }]]}
                        style={{
                          backgroundColor: 'transparent',
                          color: 'inherit',
                          fontSize: 'inherit',
                          maxWidth: '75ch', // Restrict to 80 characters per line
                          width: '100%',
                        }}
                        className="text-[14px] sm:text-[15px] [&_pre]:overflow-x-auto [&_pre]:p-3 [&_code]:text-sm [&_p]:break-words [&_p]:whitespace-pre-wrap"
                        wrapperElement={{
                          'data-color-mode': 'light'
                        }}
                      />
                    </div>
                  </Card>
                </div>
              )
            ))}

            {/* Loading Indicator */}
            {isLoading && (
              <div className="flex">
                <Card className="p-4 rounded-2xl border-2 border-border bg-secondary-background shadow-shadow">
                  <div className="flex items-center gap-1">
                    <div className="w-2 h-2 rounded-full bg-foreground/40 animate-[bounce_1.4s_infinite_.2s]" />
                    <div className="w-2 h-2 rounded-full bg-foreground/40 animate-[bounce_1.4s_infinite_.4s]" />
                    <div className="w-2 h-2 rounded-full bg-foreground/40 animate-[bounce_1.4s_infinite_.6s]" />
                  </div>
                </Card>
              </div>
            )}
          </div>
        </ScrollArea>

        {/* Add Scroll to Bottom Button */}
        <div className="relative">
          {showScrollButton && (
            <Button
              size="icon"
              variant="noShadow"
              className="absolute mx-6 bottom-4 right-4 h-10 w-10 rounded-full shadow-lg bg-main hover:bg-main/90 text-main-foreground z-50"
              onClick={scrollToBottom}
            >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="M12 19V5" />
              <path d="m5 12 7 7 7-7" />
            </svg>
          </Button>
          )}
        </div>

        {/* Input Area */}
        <div className="py-2 bg-background px-4 sm:px-8">
          <div className="flex gap-2 sm:gap-3 items-center justify-end">
            <div className="flex-1">
              <Textarea
                ref={inputRef}
                placeholder={
                  isProcessing ? "Processing repository..."
                  : isLoading ? "Waiting for response..."
                  : "Ask me anything about this repository..."
                }
                value={inputValue}
                onChange={(e) => setInputValue(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' && e.shiftKey) {
                    e.preventDefault();
                    setInputValue(prev => prev + '\n');
                  } else if (e.key === 'Enter') {
                    handleSend();
                  }
                }}
                className="text-[14px] sm:text-[15px] rounded-2xl border-2 border-border bg-secondary-background shadow-shadow text-foreground"
                style={{ height: '40px' }}
                disabled={!isConnected || isLoading || isProcessing}
              />
            </div>
            <div>
              <Button
                size="icon" 
                className="bg-main hover:bg-main/90 text-main-foreground h-12 w-12 sm:h-14 sm:w-14 rounded-2xl shadow-shadow flex items-center justify-center cursor-pointer"
                onClick={handleSend}
                disabled={isLoading || !isConnected || !inputValue.trim() || isProcessing}
              >
                <SendHorizontal className="h-5 w-5 sm:h-6 sm:w-6" />
              </Button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}