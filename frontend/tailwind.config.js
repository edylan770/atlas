/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        /** Tista navy — headers, sidebar chrome, dark UI */
        navy: {
          50: "#f4f7fb",
          100: "#e2e9f3",
          200: "#c5d4e8",
          300: "#94aed0",
          400: "#5f84b0",
          500: "#3d6394",
          600: "#2d4d78",
          700: "#1f3a5c",
          800: "#152a45",
          900: "#0b1f3a",
          950: "#071525",
        },
        /** Tista medium blue — primary actions, links, accents */
        brand: {
          50: "#eef5fc",
          100: "#d6e8f8",
          200: "#aed0f0",
          300: "#7ab3e3",
          400: "#4a94d4",
          500: "#2b7bc4",
          600: "#2368a8",
          700: "#1d5589",
          800: "#18456f",
          900: "#133a5c",
        },
        tista: {
          navy: "#0b1f3a",
          blue: "#2b7bc4",
          black: "#111111",
          white: "#ffffff",
        },
      },
      fontFamily: {
        sans: [
          "Segoe UI",
          "system-ui",
          "-apple-system",
          "BlinkMacSystemFont",
          "sans-serif",
        ],
      },
      keyframes: {
        "atlas-fade-in": {
          from: { opacity: "0", transform: "translateY(4px)" },
          to: { opacity: "1", transform: "translateY(0)" },
        },
        "atlas-thumbnail-drift": {
          "0%": {
            transform: "translate(-50%, -50%) scale(0.55)",
            opacity: "0",
          },
          "18%": { opacity: "1" },
          "100%": {
            transform: "translate(var(--drift-x), var(--drift-y)) scale(1)",
            opacity: "0.88",
          },
        },
        "atlas-soft-pulse": {
          "0%, 100%": { opacity: "0.18" },
          "50%": { opacity: "0.28" },
        },
      },
      animation: {
        "atlas-fade-in": "atlas-fade-in 0.45s ease-out forwards",
        "atlas-thumbnail-drift": "atlas-thumbnail-drift 5s ease-out infinite",
        "atlas-soft-pulse": "atlas-soft-pulse 4s ease-in-out infinite",
      },
    },
  },
  plugins: [],
};
