import { useEffect, useRef, useState, type CSSProperties } from "react";
import tistaLogoUrl from "../assets/tista-logo.png";

const LOADING_MESSAGES = [
  "Analyzing assets…",
  "Generating embeddings…",
  "Finding similar content…",
  "Ranking results…",
] as const;

const THUMBNAILS = [
  { driftX: "-130px", driftY: "-85px", delay: "0s", hue: "from-brand-400/50 to-navy-700/70" },
  { driftX: "120px", driftY: "-90px", delay: "0.35s", hue: "from-brand-300/45 to-navy-800/65" },
  { driftX: "-125px", driftY: "75px", delay: "0.7s", hue: "from-navy-400/45 to-brand-600/40" },
  { driftX: "130px", driftY: "80px", delay: "1.05s", hue: "from-brand-500/35 to-navy-900/55" },
  { driftX: "0px", driftY: "-120px", delay: "1.4s", hue: "from-brand-200/35 to-navy-700/50" },
  { driftX: "-55px", driftY: "105px", delay: "1.75s", hue: "from-brand-400/30 to-navy-800/45" },
] as const;

function NetworkBackground() {
  const nodes = [
    [12, 18], [28, 42], [45, 12], [62, 35], [78, 22], [88, 55],
    [18, 72], [35, 88], [55, 68], [72, 82], [8, 48], [92, 38],
  ];
  const edges: [number, number][] = [
    [0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 10], [10, 6], [6, 7], [7, 8], [8, 9], [5, 11], [11, 4],
  ];

  return (
    <svg
      className="pointer-events-none absolute inset-0 h-full w-full atlas-network-pulse opacity-[0.24]"
      aria-hidden
      preserveAspectRatio="xMidYMid slice"
    >
      {edges.map(([a, b], i) => {
        const [x1, y1] = nodes[a]!;
        const [x2, y2] = nodes[b]!;
        return (
          <line
            key={i}
            x1={`${x1}%`}
            y1={`${y1}%`}
            x2={`${x2}%`}
            y2={`${y2}%`}
            stroke="currentColor"
            strokeWidth="0.5"
            className="text-brand-400/60"
          />
        );
      })}
      {nodes.map(([cx, cy], i) => (
        <circle key={i} cx={`${cx}%`} cy={`${cy}%`} r="1.2" className="fill-brand-300/70" />
      ))}
    </svg>
  );
}

function FlowGraphic() {
  return (
    <div className="relative mx-auto mb-10 h-44 w-full max-w-lg overflow-visible">
      {THUMBNAILS.map((thumb, i) => (
        <div
          key={i}
          className="absolute left-1/2 top-1/2 z-[5] h-14 w-[4.5rem] atlas-thumbnail-drift rounded-lg border border-white/20 shadow-lg"
          style={
            {
              "--drift-x": `calc(-50% + ${thumb.driftX})`,
              "--drift-y": `calc(-50% + ${thumb.driftY})`,
              animationDelay: thumb.delay,
            } as CSSProperties
          }
        >
          <div className={`h-full w-full rounded-lg bg-gradient-to-br ${thumb.hue}`} />
        </div>
      ))}

      <div className="absolute left-[6%] top-1/2 z-[8] -translate-y-1/2 rounded-lg border border-white/10 bg-white/5 p-2.5 shadow-lg backdrop-blur-sm atlas-slide-in-left">
        <svg className="h-10 w-14 text-brand-300/90" viewBox="0 0 56 40" fill="none" aria-hidden>
          <rect x="4" y="4" width="48" height="32" rx="3" stroke="currentColor" strokeWidth="1.5" />
          <path
            d="M12 28V18l8 6 8-10 8 14"
            stroke="currentColor"
            strokeWidth="1.5"
            strokeLinecap="round"
            strokeLinejoin="round"
          />
          <line x1="12" y1="12" x2="28" y2="12" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" opacity="0.6" />
        </svg>
      </div>

      <div className="absolute left-1/2 top-1/2 z-10 flex h-[4.5rem] w-[4.5rem] -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full border border-brand-400/50 bg-brand-500/15 shadow-[0_0_28px_rgba(43,123,196,0.35)] atlas-hub-pulse">
        <svg className="h-9 w-9 text-brand-200" viewBox="0 0 32 32" fill="none" aria-hidden>
          <circle cx="16" cy="16" r="10" stroke="currentColor" strokeWidth="1.2" opacity="0.55" />
          <circle cx="12" cy="14" r="1.5" fill="currentColor" />
          <circle cx="20" cy="14" r="1.5" fill="currentColor" />
          <circle cx="16" cy="20" r="1.5" fill="currentColor" />
          <path d="M12 14l4 3 4-3M16 17v3" stroke="currentColor" strokeWidth="1" strokeLinecap="round" />
        </svg>
      </div>

      <div className="absolute right-[4%] top-1/2 z-[8] flex -translate-y-1/2 gap-1 atlas-slide-in-right">
        {[0, 1, 2].map((i) => (
          <div
            key={i}
            className="h-12 w-[3.25rem] rounded-md border border-white/15 bg-gradient-to-br from-brand-400/30 to-navy-800/55 shadow-md"
            style={{ transform: `rotate(${(i - 1) * 10}deg) translateY(${i === 1 ? -6 : 4}px)` }}
          />
        ))}
      </div>

      <svg className="pointer-events-none absolute inset-0 z-[6] h-full w-full text-brand-400/45" aria-hidden>
        <defs>
          <marker id="atlas-arrow" markerWidth="6" markerHeight="6" refX="5" refY="3" orient="auto">
            <path d="M0,0 L6,3 L0,6 Z" fill="currentColor" />
          </marker>
        </defs>
        <line x1="20%" y1="50%" x2="36%" y2="50%" stroke="currentColor" strokeWidth="1" markerEnd="url(#atlas-arrow)" />
        <line x1="64%" y1="50%" x2="80%" y2="50%" stroke="currentColor" strokeWidth="1" markerEnd="url(#atlas-arrow)" />
      </svg>
    </div>
  );
}

interface AtlasLoadingScreenProps {
  visible: boolean;
}

export function AtlasLoadingScreen({ visible }: AtlasLoadingScreenProps) {
  const [messageIndex, setMessageIndex] = useState(0);
  const [mounted, setMounted] = useState(visible);
  const [active, setActive] = useState(false);

  useEffect(() => {
    if (visible) {
      setMounted(true);
      setMessageIndex(0);
      const enterTimer = window.requestAnimationFrame(() => {
        window.requestAnimationFrame(() => setActive(true));
      });
      const interval = window.setInterval(() => {
        setMessageIndex((i) => (i + 1) % LOADING_MESSAGES.length);
      }, 2600);
      return () => {
        window.cancelAnimationFrame(enterTimer);
        window.clearInterval(interval);
      };
    }

    setActive(false);
    const exitTimer = window.setTimeout(() => setMounted(false), 550);
    return () => window.clearTimeout(exitTimer);
  }, [visible]);

  if (!mounted) return null;

  const progress = ((messageIndex + 1) / LOADING_MESSAGES.length) * 100;

  return (
    <div
      className={`fixed inset-0 z-[100] flex flex-col items-center justify-center bg-navy-950 transition-opacity duration-500 ease-out ${
        active ? "opacity-100" : "opacity-0"
      }`}
      role="status"
      aria-live="polite"
      aria-label="Loading"
    >
      <NetworkBackground />
      <div
        className={`relative z-10 flex w-full max-w-lg flex-col items-center px-6 transition-all duration-700 ease-out ${
          active ? "scale-100 opacity-100" : "scale-[0.96] opacity-0"
        }`}
      >
        <FlowGraphic />

        <h1
          className={`text-4xl font-bold tracking-[0.22em] text-white sm:text-5xl ${
            active ? "atlas-title-enter" : ""
          }`}
        >
          ATLAS
        </h1>
        <p className="mt-2 text-sm font-medium text-brand-300 sm:text-base">
          AI-Powered Asset Discovery
        </p>

        <div className="mt-8 w-full max-w-xs">
          <div className="h-1.5 overflow-hidden rounded-full bg-navy-800 ring-1 ring-white/10">
            <div
              className="h-full rounded-full bg-brand-500 transition-all duration-700 ease-out"
              style={{ width: `${progress}%` }}
            />
          </div>
          <p key={messageIndex} className="mt-4 atlas-message-fade text-center text-sm text-white/85">
            {LOADING_MESSAGES[messageIndex]}
          </p>
          <div className="mt-3 flex justify-center gap-2">
            {LOADING_MESSAGES.map((_, i) => (
              <span
                key={i}
                className={`h-1.5 w-1.5 rounded-full transition-colors duration-500 ${
                  i <= messageIndex ? "bg-brand-400" : "bg-navy-700"
                }`}
              />
            ))}
          </div>
        </div>
      </div>

      <div
        className={`absolute bottom-8 flex flex-col items-center gap-1.5 transition-opacity duration-700 delay-150 ${
          active ? "opacity-90" : "opacity-0"
        }`}
      >
        <img
          src={tistaLogoUrl}
          alt="Tista"
          className="h-7 w-auto max-w-[120px] object-contain brightness-0 invert opacity-90"
        />
        <p className="text-[11px] font-medium tracking-wide text-brand-300/80">
          Insights. Powered by AI.
        </p>
      </div>
    </div>
  );
}

/** Keep the overlay visible for at least minMs once shown. */
export function useMinDurationLoading(active: boolean, minMs = 3000): boolean {
  const [shown, setShown] = useState(active);
  const shownAtRef = useRef<number | null>(null);

  useEffect(() => {
    if (active) {
      shownAtRef.current = Date.now();
      setShown(true);
      return;
    }

    if (shownAtRef.current === null) {
      setShown(false);
      return;
    }

    const elapsed = Date.now() - shownAtRef.current;
    const remaining = Math.max(0, minMs - elapsed);
    const timer = window.setTimeout(() => {
      shownAtRef.current = null;
      setShown(false);
    }, remaining);
    return () => window.clearTimeout(timer);
  }, [active, minMs]);

  return shown;
}
