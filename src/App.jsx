import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import HomePage from './pages/HomePage';
import CoursePage from './pages/CoursePage';
import NotePage from './pages/NotePage';

function App() {
  return (
    <BrowserRouter>
      <div className="flex min-h-screen bg-gray-50">
        <Sidebar />
        <main className="flex-1 ml-20">
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/courses" element={<Navigate to="/" replace />} />
            <Route path="/courses/:courseId" element={<CoursePage />} />
            <Route path="/courses/:courseId/:chapterId/:noteId" element={<NotePage />} />
            <Route path="/profile" element={<div className="p-8">Profile Page (Coming Soon)</div>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  );
}

export default App;
