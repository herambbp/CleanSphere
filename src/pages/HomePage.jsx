import { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { BookOpen, TrendingUp, Clock } from 'lucide-react';
import { fetcher } from '../utils/fetcher';

export default function HomePage() {
  const [courses, setCourses] = useState([]);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    fetcher.getCourses()
      .then(setCourses)
      .catch(console.error)
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-screen">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-500"></div>
      </div>
    );
  }

  return (
    <div className="max-w-7xl mx-auto px-8 py-12">
      <div className="mb-12">
        <h1 className="text-4xl font-bold text-gray-900 mb-2">Welcome Back!</h1>
        <p className="text-gray-600">Continue your learning journey</p>
      </div>

      {/* Stats */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-12">
        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center">
              <BookOpen className="w-6 h-6 text-primary-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">{courses.length}</div>
              <div className="text-sm text-gray-600">Active Courses</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-green-100 rounded-lg flex items-center justify-center">
              <TrendingUp className="w-6 h-6 text-green-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">
                {courses.reduce((acc, c) => acc + c.totalNotes, 0)}
              </div>
              <div className="text-sm text-gray-600">Total Topics</div>
            </div>
          </div>
        </div>

        <div className="bg-white rounded-xl shadow-sm border border-gray-200 p-6">
          <div className="flex items-center gap-4">
            <div className="w-12 h-12 bg-yellow-100 rounded-lg flex items-center justify-center">
              <Clock className="w-6 h-6 text-yellow-600" />
            </div>
            <div>
              <div className="text-2xl font-bold text-gray-900">24h</div>
              <div className="text-sm text-gray-600">This Week</div>
            </div>
          </div>
        </div>
      </div>

      {/* Courses */}
      <div>
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Your Courses</h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
          {courses.map((course) => (
            <button
              key={course.id}
              onClick={() => navigate(`/courses/${course.id}`)}
              className="bg-white rounded-xl shadow-sm border border-gray-200 p-6 hover:shadow-md transition-shadow text-left"
            >
              <div className="w-12 h-12 bg-primary-100 rounded-lg flex items-center justify-center mb-4">
                <BookOpen className="w-6 h-6 text-primary-600" />
              </div>
              <h3 className="text-xl font-bold text-gray-900 mb-2">{course.name}</h3>
              <p className="text-sm text-gray-600 mb-4">
                {course.chapters.length} chapters • {course.totalNotes} topics
              </p>
              <div className="flex items-center gap-2">
                <div className="flex-1 bg-gray-200 rounded-full h-2">
                  <div className="bg-primary-500 h-2 rounded-full" style={{ width: '35%' }} />
                </div>
                <span className="text-sm text-gray-600">35%</span>
              </div>
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}
