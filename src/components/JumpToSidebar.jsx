import { useState, useEffect } from 'react';
import { Circle, CheckCircle, Clock } from 'lucide-react';

export default function JumpToSidebar({ headings, status = 'not-started' }) {
  const [activeId, setActiveId] = useState('');

  useEffect(() => {
    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            setActiveId(entry.target.id);
          }
        });
      },
      { rootMargin: '-20% 0px -80% 0px' }
    );

    headings.forEach(({ id }) => {
      const element = document.getElementById(id);
      if (element) observer.observe(element);
    });

    return () => observer.disconnect();
  }, [headings]);

  const scrollToHeading = (id) => {
    const element = document.getElementById(id);
    if (element) {
      element.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  };

  const getIcon = () => {
    switch (status) {
      case 'completed':
        return <CheckCircle className="w-4 h-4 text-green-600" />;
      case 'in-progress':
        return <Clock className="w-4 h-4 text-yellow-600" />;
      default:
        return <Circle className="w-4 h-4 text-gray-400" />;
    }
  };

  return (
    <div className="hidden xl:block fixed right-8 top-24 w-64">
      <div className="bg-white rounded-xl shadow-lg border border-gray-200 p-6">
        <div className="flex items-center gap-2 mb-4">
          {getIcon()}
          <h3 className="font-semibold text-gray-900">Jump To</h3>
        </div>

        <nav className="space-y-2">
          {headings.map((heading) => (
            <button
              key={heading.id}
              onClick={() => scrollToHeading(heading.id)}
              className={`
                w-full text-left text-sm py-1.5 px-3 rounded-lg transition-colors
                ${heading.level === 3 ? 'pl-6' : ''}
                ${
                  activeId === heading.id
                    ? 'bg-primary-50 text-primary-700 font-medium'
                    : 'text-gray-600 hover:bg-gray-50 hover:text-gray-900'
                }
              `}
            >
              {heading.text}
            </button>
          ))}
        </nav>
      </div>
    </div>
  );
}
