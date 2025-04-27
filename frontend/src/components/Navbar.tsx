import { Link } from "react-router-dom"
import { useGithubStars } from "../hooks/useGithubStars"

export function Navbar() {
  const { stars, loading, error } = useGithubStars("HarishChandran3304", "TTG")

  // Log any errors to help with debugging
  if (error) {
    console.error("Error fetching GitHub stars:", error)
  }

  return (
    <div className="w-full bg-white">
      <div className="max-w-screen-xl mx-auto px-4 sm:px-8 py-4 sm:py-6">
        <div className="flex justify-between items-center">
          {/* Logo */}
          <Link to="/" className="text-2xl sm:text-3xl font-bold hover:opacity-90 transition-opacity flex items-center gap-2">
            <div>
              <span className="text-main">TalkTo</span>
              <span className="text-foreground">GitHub</span>
            </div>
            <span className="text-[10px] px-1.5 py-0.5 rounded bg-main/10 text-main font-medium">BETA</span>
          </Link>

          {/* Social Links */}
          <div className="flex items-center gap-4">
            <a
              href="https://github.com/HarishChandran3304/TTG"
              target="_blank"
              rel="noopener noreferrer"
              className="text-foreground/80 hover:text-foreground transition-colors flex items-center gap-2 sm:gap-4"
            >
              <svg height="24" width="24" viewBox="0 0 16 16" fill="currentColor" className="sm:h-7 sm:w-7">
                <path d="M8 0c4.42 0 8 3.58 8 8a8.013 8.013 0 0 1-5.45 7.59c-.4.08-.55-.17-.55-.38 0-.27.01-1.13.01-2.2 0-.75-.25-1.23-.54-1.48 1.78-.2 3.65-.88 3.65-3.95 0-.88-.31-1.59-.82-2.15.08-.2.36-1.02-.08-2.12 0 0-.67-.22-2.2.82-.64-.18-1.32-.27-2-.27-.68 0-1.36.09-2 .27-1.53-1.03-2.2-.82-2.2-.82-.44 1.1-.16 1.92-.08 2.12-.51.56-.82 1.28-.82 2.15 0 3.06 1.86 3.75 3.64 3.95-.23.2-.44.55-.51 1.07-.46.21-1.61.55-2.33-.66-.15-.24-.6-.83-1.23-.82-.67.01-.27.38.01.53.34.19.73.9.82 1.13.16.45.68 1.31 2.69.94 0 .67.01 1.3.01 1.49 0 .21-.15.45-.55.38A7.995 7.995 0 0 1 0 8c0-4.42 3.58-8 8-8Z"/>
              </svg>
              {loading ? (
                <span className="text-sm sm:text-base">Loading...</span>
              ) : stars !== null ? (
                <span className="flex items-center gap-1 sm:gap-2">
                  <svg width="16" height="16" viewBox="0 0 16 16" fill="#FFD700" className="sm:h-5 sm:w-5">
                    <path d="M8 .25a.75.75 0 0 1 .673.418l1.882 3.815 4.21.612a.75.75 0 0 1 .416 1.279l-3.046 2.97.719 4.192a.75.75 0 0 1-1.088.791L8 12.347l-3.766 1.98a.75.75 0 0 1-1.088-.79l.72-4.194L.818 6.374a.75.75 0 0 1 .416-1.28l4.21-.611L7.327.668A.75.75 0 0 1 8 .25z"/>
                  </svg>
                  <span className="text-sm sm:text-base">{stars}</span>
                </span>
              ) : null}
            </a>
            {/* Product Hunt badge */}
            <a href="https://www.producthunt.com/posts/talktogithub?embed=true&utm_source=badge-featured&utm_medium=badge&utm_souce=badge-talktogithub" target="_blank" rel="noopener noreferrer" aria-label="View TalkToGitHub on Product Hunt">
              <img src="https://api.producthunt.com/widgets/embed-image/v1/featured.svg?post_id=957930&theme=light&t=1745739901153" alt="TalkToGitHub - Turn GitHub repositories into conversations | Product Hunt" style={{ width: 180, height: 40 }} width={180} height={40} />
            </a>
          </div>
        </div>
      </div>
      <div className="h-[4px] bg-black" />
    </div>
  )
}
