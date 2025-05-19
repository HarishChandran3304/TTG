import { Button } from "./ui/button"

interface ExampleRepo {
  name: string
  url: string
}

const EXAMPLE_REPOS: ExampleRepo[] = [
  { name: "yeoshin-backend-v2_backend", url: "https://github.com/fastlane-dev/yeoshin-backend-v2/backend" },
  { name: "yeoshin-backend-v2_admin", url: "https://github.com/fastlane-dev/yeoshin-backend-v2/admin" },
]

interface ExampleReposProps {
  onSelect: (url: string) => void
}

export function ExampleRepos({ onSelect }: ExampleReposProps) {
  return (
    <div className="flex flex-wrap gap-2 max-w-full">
      {EXAMPLE_REPOS.map((repo) => (
        <Button
          key={repo.url}
          variant="noShadow"
          size="sm"
          onClick={() => onSelect(repo.url)}
          className="bg-main hover:bg-main/50 cursor-pointer text-xs sm:text-sm py-1 px-2 sm:px-3"
        >
          {repo.name}
        </Button>
      ))}
    </div>
  )
}