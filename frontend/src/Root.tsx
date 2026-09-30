import { lazy, Suspense } from "react";
import { BrowserRouter, Route, Routes } from "react-router-dom";
import App from "./App";

const AdminApp = lazy(() => import("./admin/AdminApp"));
const DeckSuggestPage = lazy(() => import("./pages/DeckSuggestPage"));

export default function Root() {
  return (
    <BrowserRouter>
      <Suspense fallback={null}>
        <Routes>
          <Route path="/" element={<App />} />
          <Route path="/deck" element={<DeckSuggestPage />} />
          <Route path="/admin/*" element={<AdminApp />} />
        </Routes>
      </Suspense>
    </BrowserRouter>
  );
}
