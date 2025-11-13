import { useEffect, useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ChevronRight, BookOpen, Play } from 'lucide-react';
import { fetcher } from '../utils/fetcher';
import CourseTree from '../components/CourseTree';

export default function CoursePage() {
  const { courseId } = useParams();
  const [course, setCourse] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetcher
      .getCourse(courseId)
      .then(setCourse)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, [courseId]);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  if (!course) {
    return <div className="p-8">Course not found</div>;
  }

  // Get first incomplete note for "Continue" card
  const getNextNote = () => {
    for (const chapter of course.chapters) {
      const nextNote = chapter.notes.find((n) => n.status !== 'completed');
      if (nextNote) {
        return { chapter, note: nextNote };
      }
    }
    return null;
  };

  const nextItem = getNextNote();

  return (
    <div className="max-w-6xl mx-auto px-8 py-12">
      {/* Breadcrumb */}
      <nav className="flex items-center gap-2 text-sm text-gray-600 mb-6">
        <Link to="/" className="hover:text-primary-600">
          Home
        </Link>
        <ChevronRight className="w-4 h-4" />
        <Link to="/courses" className="hover:text-primary-600">
          Courses
        </Link>
        <ChevronRight className="w-4 h-4" />
        <span className="text-gray-900 font-medium">{course.name}</span>
      </nav>

      {/* Course Header */}
      <div className="mb-8">
        <div className="flex items-center gap-4 mb-4">
          <div className="w-16 h-16 bg-primary-100 rounded-xl flex items-center justify-center">
            <BookOpen className="w-8 h-8 text-primary-600" />
          </div>
          <div>
            <h1 className="text-4xl font-bold text-gray-900">{course.name}</h1>
            <p className="text-gray-600 mt-1">
              {course.chapters.length} chapters •{' '}
              {course.chapters.reduce((acc, ch) => acc + ch.notes.length, 0)} topics
            </p>
          </div>
        </div>
      </div>

      {/* Continue Learning Card */}
      {nextItem && (
        <div className="bg-gradient-to-r from-primary-500 to-primary-600 rounded-xl shadow-lg p-8 mb-8 text-white">
          <h2 className="text-2xl font-bold mb-2">Continue Where You Left Off</h2>
          <p className="text-primary-100 mb-6">
            {nextItem.chapter.name} • {nextItem.note.title}
          </p>
          <Link
            to={`/courses/${courseId}/${nextItem.chapter.id}/${nextItem.note.id}`}
            className="inline-flex items-center gap-2 bg-white text-primary-600 px-6 py-3 rounded-lg font-semibold hover:bg-primary-50 transition-colors"
          >
            <Play className="w-5 h-5" />
            Continue Learning
          </Link>
        </div>
      )}

      {/* Course Content */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Course Content</h2>
        <CourseTree chapters={course.chapters} courseId={courseId} />
      </div>
    </div>
  );
}
