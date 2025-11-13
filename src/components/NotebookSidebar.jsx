import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText, X } from 'lucide-react';
import { useNavigate, useParams } from 'react-router-dom';
import ProgressBadge from './ProgressBadge';

export default function NotebookSidebar({ chapters, courseId, isOpen, onClose }) {
  const [expandedChapters, setExpandedChapters] = useState(new Set(chapters.map(ch => ch.id)));
  const navigate = useNavigate();
  const params = useParams();

  const toggleChapter = (chapterId) => {
    setExpandedChapters((prev) => {
      const next = new Set(prev);
      if (next.has(chapterId)) {
        next.delete(chapterId);
      } else {
        next.add(chapterId);
      }
      return next;
    });
  };

  const handleNoteClick = (chapterId, noteId) => {
    navigate(`/courses/${courseId}/${chapterId}/${noteId}`);
    if (onClose) onClose();
  };

  return (
    <>
      {/* Overlay for mobile */}
      {isOpen && (
        <div
          className="lg:hidden fixed inset-0 bg-black bg-opacity-50 z-40"
          onClick={onClose}
        />
      )}

      {/* Sidebar */}
      <div
        className={`
          fixed lg:sticky top-0 h-screen w-80 bg-white border-r border-gray-200
          overflow-y-auto z-50 transition-transform duration-300
          ${isOpen ? 'translate-x-0' : '-translate-x-full lg:translate-x-0'}
        `}
      >
        <div className="p-6">
          <div className="flex items-center justify-between mb-6">
            <h2 className="text-xl font-bold text-gray-900">Course Content</h2>
            <button onClick={onClose} className="lg:hidden">
              <X className="w-5 h-5" />
            </button>
          </div>

          <div className="space-y-3">
            {chapters.map((chapter) => {
              const isExpanded = expandedChapters.has(chapter.id);

              return (
                <div key={chapter.id} className="border border-gray-200 rounded-lg overflow-hidden">
                  <button
                    onClick={() => toggleChapter(chapter.id)}
                    className="w-full px-4 py-3 flex items-center gap-2 hover:bg-gray-50"
                  >
                    {isExpanded ? (
                      <ChevronDown className="w-4 h-4 text-gray-600" />
                    ) : (
                      <ChevronRight className="w-4 h-4 text-gray-600" />
                    )}
                    <span className="text-sm font-semibold text-gray-900">{chapter.name}</span>
                  </button>

                  {isExpanded && (
                    <div className="bg-gray-50">
                      {chapter.notes.map((note) => {
                        const isActive =
                          params.chapterId === chapter.id && params.noteId === note.id;

                        return (
                          <button
                            key={note.id}
                            onClick={() => handleNoteClick(chapter.id, note.id)}
                            className={`
                              w-full px-4 py-3 flex items-start gap-2 text-left
                              border-t border-gray-200 hover:bg-white transition-colors
                              ${isActive ? 'bg-primary-50 border-l-4 border-l-primary-500' : ''}
                            `}
                          >
                            <FileText className="w-4 h-4 text-gray-400 mt-0.5 flex-shrink-0" />
                            <div className="flex-1 min-w-0">
                              <div className="text-sm font-medium text-gray-800 truncate">
                                {note.title}
                              </div>
                              <div className="mt-1">
                                <ProgressBadge status={note.status} />
                              </div>
                            </div>
                          </button>
                        );
                      })}
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        </div>
      </div>
    </>
  );
}
