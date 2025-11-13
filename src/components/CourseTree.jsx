import { useState } from 'react';
import { ChevronDown, ChevronRight, FileText } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import ProgressBadge from './ProgressBadge';

export default function CourseTree({ chapters, courseId }) {
  const [expandedChapters, setExpandedChapters] = useState(new Set([chapters[0]?.id]));
  const navigate = useNavigate();

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
  };

  return (
    <div className="space-y-4">
      {chapters.map((chapter) => {
        const isExpanded = expandedChapters.has(chapter.id);

        return (
          <div key={chapter.id} className="bg-white rounded-xl shadow-sm border border-gray-200 overflow-hidden">
            {/* Chapter Header */}
            <button
              onClick={() => toggleChapter(chapter.id)}
              className="w-full px-6 py-4 flex items-center justify-between hover:bg-gray-50 transition-colors"
            >
              <div className="flex items-center gap-3">
                {isExpanded ? (
                  <ChevronDown className="w-5 h-5 text-gray-600" />
                ) : (
                  <ChevronRight className="w-5 h-5 text-gray-600" />
                )}
                <h3 className="text-lg font-semibold text-gray-900">{chapter.name}</h3>
                <span className="text-sm text-gray-500">({chapter.notes.length} topics)</span>
              </div>
            </button>

            {/* Chapter Notes */}
            {isExpanded && (
              <div className="border-t border-gray-100 bg-gray-50">
                {chapter.notes.map((note) => (
                  <button
                    key={note.id}
                    onClick={() => handleNoteClick(chapter.id, note.id)}
                    className="w-full px-6 py-4 flex items-center justify-between hover:bg-white transition-colors border-b border-gray-100 last:border-b-0"
                  >
                    <div className="flex items-center gap-3">
                      <FileText className="w-4 h-4 text-gray-400" />
                      <span className="text-gray-800 font-medium">{note.title}</span>
                      <span className="text-sm text-gray-500">{note.readTime}</span>
                    </div>
                    <ProgressBadge status={note.status} />
                  </button>
                ))}
              </div>
            )}
          </div>
        );
      })}
    </div>
  );
}
