import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ChevronRight, Clock, CheckCircle, Menu, Edit3 } from 'lucide-react';
import { fetcher } from '../utils/fetcher';
import { extractHeadings } from '../utils/calculateReadTime';
import MarkdownRenderer from '../components/MarkdownRenderer';
import JumpToSidebar from '../components/JumpToSidebar';
import NotebookSidebar from '../components/NotebookSidebar';

export default function NotePage() {
  const { courseId, chapterId, noteId } = useParams();
  const [note, setNote] = useState(null);
  const [course, setCourse] = useState(null);
  const [headings, setHeadings] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [isEditing, setIsEditing] = useState(false);
  const [editContent, setEditContent] = useState('');

  useEffect(() => {
    Promise.all([
      fetcher.getNote(courseId, chapterId, noteId),
      fetcher.getCourse(courseId),
    ])
      .then(([noteData, courseData]) => {
        setNote(noteData);
        setCourse(courseData);
        setHeadings(extractHeadings(noteData.content));
        setEditContent(noteData.content);
      })
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [courseId, chapterId, noteId]);

  const toggleStatus = async () => {
    const newStatus = note.status === 'completed' ? 'in-progress' : 'completed';
    await fetcher.updateNoteStatus(courseId, chapterId, noteId, newStatus);
    setNote({ ...note, status: newStatus });
  };

  const handleSave = async () => {
    await fetcher.saveNote(courseId, chapterId, noteId, editContent);
    setNote({ ...note, content: editContent });
    setHeadings(extractHeadings(editContent));
    setIsEditing(false);
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!note || !course) {
    return <div className="p-8">Note not found</div>;
  }

  const currentChapter = course.chapters.find((ch) => ch.id === chapterId);

  return (
    <div className="flex">
      <NotebookSidebar
        chapters={course.chapters}
        courseId={courseId}
        isOpen={sidebarOpen}
        onClose={() => setSidebarOpen(false)}
      />

      <div className="flex-1 min-w-0">
        <div className="max-w-4xl mx-auto px-8 py-12">
          {/* Mobile menu button */}
          <button
            onClick={() => setSidebarOpen(true)}
            className="lg:hidden fixed bottom-6 left-6 w-14 h-14 bg-primary-500 text-white rounded-full shadow-lg flex items-center justify-center z-30"
          >
            <Menu className="w-6 h-6" />
          </button>

          {/* Breadcrumb */}
          <nav className="flex items-center gap-2 text-sm text-gray-600 mb-6 flex-wrap">
            <Link to="/" className="hover:text-primary-600">Home</Link>
            <ChevronRight className="w-4 h-4" />
            <Link to={`/courses/${courseId}`} className="hover:text-primary-600">
              {course.name}
            </Link>
            <ChevronRight className="w-4 h-4" />
            <span className="text-gray-900 font-medium">{currentChapter?.name}</span>
          </nav>

          {/* Note Header */}
          <div className="mb-8">
            <h1 className="text-4xl font-bold text-gray-900 mb-4">{note.title}</h1>
            <div className="flex flex-wrap items-center gap-4 text-gray-600">
              <div className="flex items-center gap-2">
                <Clock className="w-5 h-5" />
                <span>{note.readTime}</span>
              </div>
              <button
                onClick={toggleStatus}
                className={`
                  flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors
                  ${
                    note.status === 'completed'
                      ? 'bg-green-100 text-green-700 hover:bg-green-200'
                      : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                  }
                `}
              >
                <CheckCircle className="w-5 h-5" />
                {note.status === 'completed' ? 'Completed' : 'Mark as Complete'}
              </button>
              <button
                onClick={() => setIsEditing(!isEditing)}
                className="flex items-center gap-2 px-4 py-2 rounded-lg font-medium bg-primary-100 text-primary-700 hover:bg-primary-200 transition-colors"
              >
                <Edit3 className="w-5 h-5" />
                {isEditing ? 'Cancel' : 'Edit'}
              </button>
            </div>
          </div>

          {/* Content */}
          {isEditing ? (
            <div className="mb-8">
              <textarea
                value={editContent}
                onChange={(e) => setEditContent(e.target.value)}
                className="w-full h-96 p-4 border border-gray-300 rounded-lg font-mono text-sm"
              />
              <button
                onClick={handleSave}
                className="mt-4 px-6 py-2 bg-primary-500 text-white rounded-lg hover:bg-primary-600"
              >
                Save Changes
              </button>
            </div>
          ) : (
            <article className="bg-white rounded-xl shadow-sm border border-gray-200 p-8 mb-8">
              <MarkdownRenderer content={note.content} />
            </article>
          )}
        </div>
      </div>

      {/* Jump To Sidebar */}
      <JumpToSidebar headings={headings} status={note.status} />
    </div>
  );
}
