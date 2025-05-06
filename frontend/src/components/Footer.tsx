import config from "../config"
export function Footer() {
  return (
    <div className="w-full bg-white mt-auto">
      <div className="h-[4px] bg-black" />
      <div className="max-w-screen-xl mx-auto px-4 sm:px-8 py-4 sm:py-5">
        <div className="flex items-center justify-between w-full">
          <a
            href={config.DISCORD_INVITE_LINK}
            target="_blank"
            rel="noopener noreferrer"
            className="text-foreground/50 text-xs hover:text-main transition-colors underline underline-offset-2"
            aria-label="Get help on Discord"
          >
            Need help?
          </a>
          <div className="text-foreground/80 text-center text-xs sm:text-sm">
            Made with ❤️ by
            <a href="https://x.com/harishfelloff" target="_blank" rel="noopener noreferrer">
              <span className="text-main"> Harish</span>
            </a>
          </div>
          <a
            href="https://github.com/HarishChandran3304/TTG/issues/new"
            target="_blank"
            rel="noopener noreferrer"
            className="text-foreground/50 text-xs hover:text-main transition-colors underline underline-offset-2 text-right"
            aria-label="Report an issue on GitHub"
          >
            Report an issue
          </a>
        </div>
      </div>
    </div>
  )
}
