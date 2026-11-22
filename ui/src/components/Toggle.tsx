interface ToggleProps {
  on: boolean;
  onClick?: () => void;
}

/** iOS-style switch, green when on — as in LM Studio's Developer page. */
export default function Toggle({ on, onClick }: ToggleProps) {
  return (
    <button
      type="button"
      aria-pressed={on}
      onClick={onClick}
      className={`h-5 w-9 shrink-0 rounded-full p-0.5 transition-colors ${
        on ? "bg-ok" : "bg-edge"
      }`}
    >
      <span
        className={`block h-4 w-4 rounded-full bg-white shadow transition-transform ${
          on ? "translate-x-4" : "translate-x-0"
        }`}
      />
    </button>
  );
}
