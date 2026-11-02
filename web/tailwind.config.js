/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        heading: ['"Figtree"', "ui-sans-serif", "system-ui", "sans-serif"],
        subhead: ['"IBM Plex Mono"', "ui-monospace", "monospace"],
        sans: ['"Figtree"', "ui-sans-serif", "system-ui", "sans-serif"],
      },
      colors: {
        ink: "#07080a",
        panel: "#101318",
        accent: "#2fd3c5",
      },
    },
  },
  plugins: [],
};
