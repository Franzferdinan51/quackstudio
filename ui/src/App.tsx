import { useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Code2,
  LibraryBig,
  MessageSquare,
  Search,
  Settings as SettingsIcon,
} from "lucide-react";
import Chat from "./pages/Chat";
import Developer from "./pages/Developer";
import Library from "./pages/Library";
import Discover from "./pages/Discover";
import SettingsPage from "./pages/Settings";

type Page = "chat" | "developer" | "library" | "discover" | "settings";

const TITLES: Record<Page, string> = {
  chat: "Chat",
  developer: "Developer",
  library: "My Models",
  discover: "Discover",
  settings: "Settings",
};

interface RailItem {
  page: Page;
  label: string;
  icon: LucideIcon;
}

const RAIL_TOP: RailItem[] = [
  { page: "chat", label: "Chat", icon: MessageSquare },
  { page: "developer", label: "Developer", icon: Code2 },
  { page: "library", label: "Library", icon: LibraryBig },
  { page: "discover", label: "Discover", icon: Search },
];

function RailButton({
  item,
  active,
  onSelect,
}: {
  item: RailItem;
  active: boolean;
  onSelect: (page: Page) => void;
}) {
  const Icon = item.icon;
  return (
    <button
      type="button"
      title={item.label}
      aria-label={item.label}
      aria-current={active ? "page" : undefined}
      onClick={() => onSelect(item.page)}
      className={`relative flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
        active
          ? "bg-panel text-ink"
          : "text-muted hover:bg-panel hover:text-ink"
      }`}
    >
      {active && (
        <span className="absolute left-[-10px] h-5 w-[3px] rounded-full bg-accent" />
      )}
      <Icon size={19} className={active ? "text-accent" : undefined} />
    </button>
  );
}

export default function App() {
  const [page, setPage] = useState<Page>("chat");

  return (
    <div className="flex h-full bg-app-bg text-ink">
      {/* Left icon rail */}
      <nav className="flex w-14 shrink-0 flex-col items-center gap-1.5 border-r border-edge py-3">
        {RAIL_TOP.map((item) => (
          <RailButton
            key={item.page}
            item={item}
            active={page === item.page}
            onSelect={setPage}
          />
        ))}
        <div className="mt-auto">
          <RailButton
            item={{ page: "settings", label: "Settings", icon: SettingsIcon }}
            active={page === "settings"}
            onSelect={setPage}
          />
        </div>
      </nav>

      {/* Main column: title bar + page */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex h-11 shrink-0 items-center justify-center border-b border-edge">
          <span className="text-[13px] font-medium">{TITLES[page]}</span>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          {page === "chat" && <Chat />}
          {page === "developer" && <Developer />}
          {page === "library" && <Library />}
          {page === "discover" && <Discover />}
          {page === "settings" && <SettingsPage />}
        </main>
      </div>
    </div>
  );
}
