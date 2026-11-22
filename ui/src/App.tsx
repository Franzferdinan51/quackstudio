import { useCallback, useEffect, useMemo, useState } from "react";
import type { LucideIcon } from "lucide-react";
import {
  Code2,
  Cpu,
  Download,
  FolderPlus,
  LibraryBig,
  MemoryStick,
  MessageSquare,
  MoreHorizontal,
  Network,
  PanelLeft,
  PanelRight,
  Search,
  Settings as SettingsIcon,
  SquarePen,
} from "lucide-react";
import Chat from "./pages/Chat";
import Developer from "./pages/Developer";
import Library from "./pages/Library";
import Discover from "./pages/Discover";
import LinkPage from "./pages/Link";
import SettingsPage from "./pages/Settings";
import DownloadsPane from "./components/DownloadsPane";
import { apiBase } from "./api";
import type { LibraryCategory, StatusView } from "./types";
import { formatGB } from "./types";
import type { ChatsState } from "./store";
import {
  loadChats,
  loadPrefs,
  newChatSession,
  saveChats,
} from "./store";

export type Page =
  | "chat"
  | "developer"
  | "library"
  | "link"
  | "discover"
  | "settings";

const TITLES: Record<Page, string> = {
  chat: "Chat",
  developer: "Model Server",
  library: "My Models",
  link: "LM Link",
  discover: "Model Search",
  settings: "Settings",
};

interface RailItem {
  page: Page;
  label: string;
  icon: LucideIcon;
}

const NAV: RailItem[] = [
  { page: "chat", label: "Chat", icon: MessageSquare },
  { page: "developer", label: "Model Server", icon: Code2 },
  { page: "library", label: "Library", icon: LibraryBig },
  { page: "link", label: "LM Link", icon: Network },
  { page: "discover", label: "Model Search", icon: Search },
];

const LIBRARY_CATEGORIES: { id: LibraryCategory; label: string }[] = [
  { id: "all", label: "View All" },
  { id: "llms", label: "LLMs" },
  { id: "embedding", label: "Text Embedding" },
  { id: "decision", label: "Decision Models" },
  { id: "drafters", label: "Drafters" },
];

const REST_LINKS: { label: string; method?: "GET" | "POST" }[] = [
  { label: "Overview" },
  { label: "Quickstart" },
  { label: "Stateful Chats" },
  { label: "Streaming events" },
  { label: "Chat with a model", method: "POST" },
  { label: "List your models", method: "GET" },
  { label: "Load a model", method: "POST" },
  { label: "Download a model", method: "POST" },
  { label: "Unload a model", method: "POST" },
  { label: "Get download status", method: "GET" },
  { label: "REST API v0" },
];

export default function App() {
  const [page, setPage] = useState<Page>("chat");
  const [expanded, setExpanded] = useState(false);
  const [status, setStatus] = useState<StatusView | null>(null);
  const [downloadsOpen, setDownloadsOpen] = useState(false);
  const [chatPanelOpen, setChatPanelOpen] = useState(true);
  const [libraryCategory, setLibraryCategory] =
    useState<LibraryCategory>("all");
  const [chatsState, setChatsState] = useState<ChatsState>(() => loadChats());
  const [chatSearch, setChatSearch] = useState("");
  const prefs = useMemo(loadPrefs, []);

  // Persist chats + apply the Appearance theme choice.
  useEffect(() => {
    saveChats(chatsState);
  }, [chatsState]);
  useEffect(() => {
    const light =
      prefs.theme === "light" ||
      (prefs.theme === "auto" &&
        window.matchMedia?.("(prefers-color-scheme: light)").matches);
    document.documentElement.classList.toggle("light", !!light);
  }, [prefs.theme]);

  // Live status for the sidebar pills / LM Link / Settings.
  const fetchStatus = useCallback(async () => {
    try {
      const base = await apiBase();
      const res = await fetch(`${base}/api/v1/status`);
      if (res.ok) setStatus((await res.json()) as StatusView);
    } catch {
      // Daemon down — pills just stay at their last values.
    }
  }, []);
  useEffect(() => {
    void fetchStatus();
    const timer = setInterval(() => void fetchStatus(), 5000);
    return () => clearInterval(timer);
  }, [fetchStatus]);

  // Library's "Use in New Chat" + Downloads' "Load Model" hand off
  // through window events (the model key rides in sessionStorage).
  useEffect(() => {
    const onUseInChat = () => setPage("chat");
    const onOpenLoad = () => setPage("library");
    window.addEventListener("quack:use-in-chat", onUseInChat);
    window.addEventListener("quack:open-load", onOpenLoad);
    return () => {
      window.removeEventListener("quack:use-in-chat", onUseInChat);
      window.removeEventListener("quack:open-load", onOpenLoad);
    };
  }, []);

  const activeChat =
    chatsState.chats.find((c) => c.id === chatsState.activeId) ??
    chatsState.chats[0];

  function newChat() {
    const chat = newChatSession();
    setChatsState((prev) => ({
      ...prev,
      chats: [chat, ...prev.chats],
      activeId: chat.id,
    }));
    setPage("chat");
  }

  function newFolder() {
    const name = `Folder ${chatsState.folders.length + 1}`;
    const id = `folder-${Date.now()}`;
    setChatsState((prev) => ({
      ...prev,
      folders: [...prev.folders, { id, name }],
    }));
  }

  function openDownloadedModel(modelKey: string) {
    sessionStorage.setItem("quack.loadModel", modelKey);
    setDownloadsOpen(false);
    setPage("library");
    window.dispatchEvent(new Event("quack:open-load"));
  }

  const sortedChats = useMemo(() => {
    const list = [...chatsState.chats];
    if (prefs.chatSort === "created-asc") {
      list.sort((a, b) => a.createdAt - b.createdAt);
    } else if (prefs.chatSort === "name") {
      list.sort((a, b) => a.name.localeCompare(b.name));
    } else {
      list.sort((a, b) => b.createdAt - a.createdAt);
    }
    const needle = chatSearch.trim().toLowerCase();
    return needle
      ? list.filter((c) => c.name.toLowerCase().includes(needle))
      : list;
  }, [chatsState.chats, prefs.chatSort, chatSearch]);

  const loadedGB = status ? formatGB(status.host.loaded_bytes) : "0.00 GB";
  const rssMB = status
    ? `${Math.round(status.host.daemon_rss_bytes / 1_000_000)} MB`
    : "—";

  return (
    <div className="flex h-full bg-app-bg text-ink">
      {/* Sidebar: icon rail or expanded labeled mode */}
      <nav
        className={`flex shrink-0 flex-col border-r border-edge py-3 ${
          expanded ? "w-64 px-2.5" : "w-14 items-center gap-1.5"
        }`}
      >
        <div
          className={`flex items-center ${expanded ? "justify-between px-1" : "justify-center"}`}
        >
          {expanded && (
            <span className="text-[13px] font-semibold">QuackStudio</span>
          )}
          <button
            type="button"
            title={expanded ? "Collapse sidebar" : "Expand sidebar"}
            onClick={() => setExpanded((v) => !v)}
            className="flex h-8 w-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel hover:text-ink"
          >
            <PanelLeft size={16} />
          </button>
        </div>

        {/* Main nav */}
        <div
          className={`mt-2 flex flex-col ${expanded ? "gap-0.5" : "items-center gap-1.5"}`}
        >
          {NAV.map((item) => {
            const Icon = item.icon;
            const active = page === item.page;
            return expanded ? (
              <button
                key={item.page}
                type="button"
                onClick={() => setPage(item.page)}
                aria-current={active ? "page" : undefined}
                className={`flex items-center gap-2.5 rounded-lg px-2.5 py-2 text-left text-[13px] transition-colors ${
                  active
                    ? "bg-primary text-white"
                    : "text-ink hover:bg-panel"
                }`}
              >
                <Icon size={16} />
                {item.label}
              </button>
            ) : (
              <button
                key={item.page}
                type="button"
                title={item.label}
                aria-label={item.label}
                aria-current={active ? "page" : undefined}
                onClick={() => setPage(item.page)}
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
          })}
        </div>

        {/* Contextual section (expanded mode only) */}
        {expanded && (
          <div className="mt-4 min-h-0 flex-1 overflow-auto">
            {page === "chat" && (
              <div>
                <div className="flex items-center justify-between px-2">
                  <span className="text-[13px] font-semibold">Chats</span>
                  <span className="flex gap-1">
                    <button
                      type="button"
                      title="Chat list options"
                      className="flex h-6 w-6 items-center justify-center rounded-md text-muted hover:bg-panel hover:text-ink"
                    >
                      <MoreHorizontal size={14} />
                    </button>
                    <button
                      type="button"
                      title="New chat"
                      onClick={newChat}
                      className="flex h-6 w-6 items-center justify-center rounded-md text-muted hover:bg-panel hover:text-ink"
                    >
                      <SquarePen size={14} />
                    </button>
                  </span>
                </div>
                <div className="mt-2 flex items-center gap-2 rounded-lg border border-edge bg-panel px-2.5 py-1.5">
                  <Search size={13} className="shrink-0 text-muted" />
                  <input
                    value={chatSearch}
                    onChange={(e) => setChatSearch(e.target.value)}
                    placeholder="Search"
                    className="w-full bg-transparent text-[13px] outline-none placeholder:text-muted"
                  />
                </div>
                <button
                  type="button"
                  onClick={newFolder}
                  className="mt-2 flex items-center gap-2 rounded-lg px-2 py-1.5 text-[13px] text-muted transition-colors hover:bg-panel hover:text-ink"
                >
                  <FolderPlus size={14} />
                  New Folder
                </button>
                {chatsState.folders.map((folder) => (
                  <div key={folder.id} className="mt-1">
                    <p className="px-2 py-1 text-[12px] font-medium text-muted">
                      ▾ {folder.name}
                    </p>
                    {sortedChats
                      .filter((c) => c.folderId === folder.id)
                      .map((c) => (
                        <ChatRow
                          key={c.id}
                          name={c.name}
                          active={c.id === activeChat?.id}
                          onClick={() => {
                            setChatsState((prev) => ({
                              ...prev,
                              activeId: c.id,
                            }));
                            setPage("chat");
                          }}
                        />
                      ))}
                  </div>
                ))}
                <div className="mt-1 flex flex-col">
                  {sortedChats
                    .filter((c) => !c.folderId)
                    .map((c) => (
                      <ChatRow
                        key={c.id}
                        name={c.name}
                        active={c.id === activeChat?.id}
                        onClick={() => {
                          setChatsState((prev) => ({
                            ...prev,
                            activeId: c.id,
                          }));
                          setPage("chat");
                        }}
                      />
                    ))}
                  {sortedChats.length === 0 && (
                    <p className="px-2 py-2 text-[12px] text-muted">
                      No chats match.
                    </p>
                  )}
                </div>
              </div>
            )}

            {page === "developer" && (
              <div>
                <p className="px-2 text-[12px] font-semibold tracking-wide text-muted uppercase">
                  Developer
                </p>
                <p className="mt-1 rounded-lg bg-accent/25 px-2.5 py-1.5 text-[13px]">
                  Local Server
                </p>
                <p className="px-2.5 py-1.5 text-[13px] text-muted">
                  Developer Docs
                </p>
                <p className="mt-4 px-2 text-[12px] font-semibold tracking-wide text-muted uppercase">
                  REST API
                </p>
                <div className="mt-1 flex flex-col">
                  {REST_LINKS.map((link) => (
                    <span
                      key={link.label}
                      className="flex items-center justify-between px-2.5 py-1.5 text-[13px] text-muted"
                    >
                      {link.label}
                      {link.method && (
                        <span
                          className={`rounded border px-1 text-[10px] ${
                            link.method === "GET"
                              ? "border-ok/50 text-ok"
                              : "border-primary/50 text-primary"
                          }`}
                        >
                          {link.method}
                        </span>
                      )}
                    </span>
                  ))}
                </div>
                <p className="mt-4 px-2 text-[12px] font-semibold tracking-wide text-muted uppercase">
                  OpenAI Compatibility
                </p>
                <p className="px-2.5 py-1.5 text-[13px] text-muted">
                  Supported endpoints
                </p>
              </div>
            )}

            {page === "library" && (
              <div>
                <p className="px-2 text-[13px] font-semibold">My Models</p>
                <div className="mt-1.5 flex flex-col">
                  {LIBRARY_CATEGORIES.map((cat) => (
                    <button
                      key={cat.id}
                      type="button"
                      onClick={() => setLibraryCategory(cat.id)}
                      className={`rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors ${
                        libraryCategory === cat.id
                          ? "bg-panel-2 text-ink"
                          : "text-muted hover:bg-panel hover:text-ink"
                      }`}
                    >
                      {cat.label}
                    </button>
                  ))}
                </div>
              </div>
            )}

            {page === "link" && (
              <div>
                <p className="px-2 text-[12px] font-semibold tracking-wide text-muted uppercase">
                  This device
                </p>
                <p className="mt-1 truncate px-2.5 py-1.5 text-[13px]">
                  {status?.host.name ?? "This Mac"}
                </p>
                <p className="mt-4 px-2 text-[12px] font-semibold tracking-wide text-muted uppercase">
                  Network Devices
                </p>
                <p className="px-2.5 py-1.5 text-[13px] text-muted">
                  None — Maya providers are the linked services.
                </p>
                <div className="mt-4 rounded-xl border border-edge bg-panel px-3 py-2.5">
                  <p className="text-[13px] font-medium">Personal Network</p>
                  <p className="mt-0.5 flex items-center gap-1.5 text-[12px] text-muted">
                    <span className="h-2 w-2 rounded-full bg-ok" />
                    {status
                      ? `${status.maya_providers_online}/${status.maya_providers} Maya providers online`
                      : "Checking…"}
                  </p>
                </div>
              </div>
            )}
          </div>
        )}

        {/* Footer: settings gear + live pills */}
        <div
          className={`mt-auto ${expanded ? "" : "flex flex-col items-center gap-1.5"}`}
        >
          {expanded ? (
            <div className="flex items-center gap-2 px-1">
              <button
                type="button"
                title="Settings"
                aria-label="Settings"
                onClick={() => setPage("settings")}
                className={`flex h-8 w-8 items-center justify-center rounded-lg transition-colors ${
                  page === "settings"
                    ? "bg-panel text-ink"
                    : "text-muted hover:bg-panel hover:text-ink"
                }`}
              >
                <SettingsIcon size={16} />
              </button>
              <span
                className="flex items-center gap-1.5 rounded-lg border border-edge bg-panel px-2 py-1 text-[11px] text-muted"
                title="Loaded model weights (sum of loaded models' file sizes)"
              >
                <MemoryStick size={12} />
                {loadedGB}
              </span>
              <span
                className="flex items-center gap-1.5 rounded-lg border border-edge bg-panel px-2 py-1 text-[11px] text-muted"
                title="quackd's own peak RSS — quackd does not sample CPU%, so its memory stands in for the CPU pill"
              >
                <Cpu size={12} />
                {rssMB}
              </span>
            </div>
          ) : (
            <button
              type="button"
              title="Settings"
              aria-label="Settings"
              aria-current={page === "settings" ? "page" : undefined}
              onClick={() => setPage("settings")}
              className={`relative flex h-10 w-10 items-center justify-center rounded-lg transition-colors ${
                page === "settings"
                  ? "bg-panel text-ink"
                  : "text-muted hover:bg-panel hover:text-ink"
              }`}
            >
              {page === "settings" && (
                <span className="absolute left-[-10px] h-5 w-[3px] rounded-full bg-accent" />
              )}
              <SettingsIcon
                size={19}
                className={page === "settings" ? "text-accent" : undefined}
              />
            </button>
          )}
        </div>
      </nav>

      {/* Main column: title bar + page */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="relative flex h-11 shrink-0 items-center justify-center border-b border-edge">
          <span className="text-[13px] font-medium">{TITLES[page]}</span>
          <div className="absolute right-3 flex items-center gap-1">
            <button
              type="button"
              title="Downloads"
              onClick={() => setDownloadsOpen(true)}
              className="flex h-8 w-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel hover:text-ink"
            >
              <Download size={16} />
            </button>
            {page === "chat" && (
              <button
                type="button"
                title="Toggle right panel"
                onClick={() => setChatPanelOpen((v) => !v)}
                className="flex h-8 w-8 items-center justify-center rounded-lg text-muted transition-colors hover:bg-panel hover:text-ink"
              >
                <PanelRight size={16} />
              </button>
            )}
          </div>
        </header>
        <main className="min-h-0 flex-1 overflow-hidden">
          {page === "chat" && (
            <Chat
              chatsState={chatsState}
              onChatsState={setChatsState}
              panelOpen={chatPanelOpen}
              onNewChat={newChat}
            />
          )}
          {page === "developer" && <Developer />}
          {page === "library" && <Library category={libraryCategory} />}
          {page === "link" && <LinkPage status={status} />}
          {page === "discover" && <Discover />}
          {page === "settings" && <SettingsPage status={status} />}
        </main>
      </div>

      {downloadsOpen && (
        <DownloadsPane
          onClose={() => setDownloadsOpen(false)}
          onLoadModel={openDownloadedModel}
        />
      )}
    </div>
  );
}

function ChatRow({
  name,
  active,
  onClick,
}: {
  name: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      className={`truncate rounded-lg px-2.5 py-1.5 text-left text-[13px] transition-colors ${
        active ? "bg-panel-2 text-ink" : "text-muted hover:bg-panel hover:text-ink"
      }`}
    >
      {name}
    </button>
  );
}
