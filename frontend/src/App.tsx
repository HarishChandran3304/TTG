import { BrowserRouter as Router, Routes, Route, useNavigate } from "react-router-dom"
import { useState } from "react"
import * as HoverCard from '@radix-ui/react-hover-card'
import { Button } from "./components/ui/button"
import { Card, CardContent } from "./components/ui/card"
import { Input } from "./components/ui/input"
import { Chat } from "./components/Chat"
import { WebSocketProvider } from "./context/WebSocketContext"
import { Navbar } from "./components/Navbar"
import { Footer } from "./components/Footer"
import { ExampleRepos } from "./components/ExampleRepos"
import demoVideo from "./assets/TTG-prefix-demo.mp4"
import { NotFound } from "./components/NotFound"

function LandingPage() {
  const navigate = useNavigate()
  const [repoUrl, setRepoUrl] = useState("")
  const [error, setError] = useState("")
  const [isProcessing, setIsProcessing] = useState(false)

  const parseGithubUrl = (url: string): { owner: string; repo: string, subName?: string } | null => {
    try {
      const urlObj = new URL(url)
      if (urlObj.hostname !== 'github.com') {
        return null
      }
      const pathParts = urlObj.pathname.split('/').filter(Boolean)
      if (pathParts.length < 2) {
        return null
      }

      if (pathParts.length == 2) {
        return {
          owner: pathParts[0],
          repo: pathParts[1]
        }
      }

      return {
        owner: pathParts[0],
        repo: pathParts[1],
        subName: pathParts[2]
      }
    } catch {
      return null
    }
  }

  const handleStartChat = async () => {
    const parsed = parseGithubUrl(repoUrl)
    if (!parsed) {
      setError("Please enter a valid GitHub repository URL (e.g., https://github.com/owner/repo)")
      return
    }

    setIsProcessing(true)
    try {
      const to = parsed.subName?
          `/${parsed.owner}/${parsed.repo}/${parsed.subName}`
          : `/${parsed.owner}/${parsed.repo}`
      navigate(to)
    } catch (err) {
      console.error(err)
      setError('Failed to start chat')
    } finally {
      setIsProcessing(false)
    }
  }

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter') {
      e.preventDefault()
      handleStartChat()
    }
  }

  return (
    <main className="min-h-screen w-full flex flex-col bg-background bg-[linear-gradient(to_right,#80808033_1px,transparent_1px),linear-gradient(to_bottom,#80808033_1px,transparent_1px)] bg-[size:70px_70px] overflow-hidden relative">
      <Navbar />

      {/* Main Content */}
      <div className="flex-1 flex flex-col items-center p-4">
        <div className="pt-8 sm:pt-12 pb-16 sm:pb-20 px-4">
          <h1 className="text-4xl sm:text-5xl md:text-7xl font-extrabold text-center max-w-5xl tracking-tight">
            Repo to Convo in <span className="text-main">seconds</span>!
          </h1>
        </div>

        <div className="flex flex-col items-center justify-center w-full max-w-[800px] -mt-4 sm:-mt-8 px-4">
          <h2 className="text-lg sm:text-xl md:text-2xl font-medium text-center mb-8">
            Chat with any public GitHub repository. No more endless docs. Why read when you can ask and get answers instantly? 🚀
          </h2>

          <Card className="w-full relative mb-6">
            <div className="absolute -left-3 sm:-left-6 -top-3 sm:-top-6 w-full h-full bg-black -z-10" />
            <CardContent className="p-4 sm:p-6 py-4">
              <div className="flex flex-col gap-6 sm:gap-8">
                <div className="flex flex-col gap-6 sm:gap-8">
                  <div className="flex flex-col sm:flex-row gap-4">
                    <Input
                      type="url"
                      placeholder="https://github.com/username/repo"
                      className={`text-base sm:text-lg py-4 sm:py-6 ${error ? 'border-red-500 focus-visible:ring-red-500' : ''}`}
                      value={repoUrl}
                      onChange={(e) => {
                        setRepoUrl(e.target.value)
                        setError("") // Clear error when user types
                      }}
                      onKeyDown={handleKeyPress}
                    />
                    <Button
                      size="lg"
                      className="text-lg sm:text-xl px-6 sm:px-8 py-4 sm:py-6 whitespace-nowrap cursor-pointer"
                      onClick={handleStartChat}
                      disabled={isProcessing}
                    >
                      {isProcessing ? 'Starting...' : 'Start Chatting'}
                    </Button>
                  </div>
                  {error && (
                    <p className="text-red-500 text-sm px-1">
                      {error}
                    </p>
                  )}
                  <div className="flex flex-col items-start gap-3 w-full">
                    <p className="text-sm font-medium text-foreground/70">
                      Try these example repositories:
                    </p>
                    <ExampleRepos onSelect={setRepoUrl} />
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>

          <p className="text-base sm:text-xl text-foreground/70 text-center px-4 flex items-center justify-center gap-2">
            You can also add the "talkto" prefix to any public GitHub URL
            <HoverCard.Root>
              <HoverCard.Trigger asChild>
                <span className="cursor-progress bg-foreground/10 rounded-full w-5 h-5 inline-flex items-center justify-center text-sm font-medium hover:bg-foreground/20 transition-colors">i</span>
              </HoverCard.Trigger>
              <HoverCard.Portal>
                <HoverCard.Content className="z-50 w-[300px] sm:w-[400px] rounded-lg border-2 border-border bg-white p-2" sideOffset={5}>
                  <video
                    src={demoVideo}
                    autoPlay
                    loop
                    muted
                    playsInline
                    className="w-full rounded"
                  />
                  <HoverCard.Arrow className="fill-border" />
                </HoverCard.Content>
              </HoverCard.Portal>
            </HoverCard.Root>
          </p>
        </div>
      </div>

      <Footer />
    </main>
  )
}

function App() {
  return (
    <WebSocketProvider>
      <Router>
        <Routes>
          <Route path="/" element={<LandingPage />} />
          <Route path=":owner/:repo" element={
            <div className="flex flex-col min-h-screen">
              <Chat />
              <Footer />
            </div>
          } />
          <Route path=":owner/:repo/:subName" element={
            <div className="flex flex-col min-h-screen">
              <Chat />
              <Footer />
            </div>
          } />
          <Route path="*" element={<NotFound />} />
        </Routes>
      </Router>
    </WebSocketProvider>
  )
}

export default App
