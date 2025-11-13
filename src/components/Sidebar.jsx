import { Home, BookOpen, User } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';

export default function Sidebar() {
  const location = useLocation();

  const navItems = [
    { icon: Home, label: 'Home', path: '/' },
    { icon: BookOpen, label: 'Courses', path: '/courses' },
    { icon: User, label: 'Profile', path: '/profile' },
  ];

  const isActive = (path) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  return (
    <div className="fixed left-0 top-0 h-screen w-20 bg-sidebar-bg flex flex-col items-center py-8 border-r border-gray-800">
      {/* Logo */}
      <div className="mb-12">
        <div className="w-10 h-10 bg-primary-500 rounded-lg flex items-center justify-center">
          <BookOpen className="w-6 h-6 text-white" />
        </div>
      </div>

      {/* Navigation */}
      <nav className="flex-1 flex flex-col gap-6">
        {navItems.map((item) => {
          const Icon = item.icon;
          const active = isActive(item.path);

          return (
            <Link
              key={item.path}
              to={item.path}
              className={`
                w-12 h-12 flex items-center justify-center rounded-xl
                transition-all duration-200
                ${active
                  ? 'bg-primary-500 text-white'
                  : 'text-gray-400 hover:bg-sidebar-hover hover:text-white'
                }
              `}
              title={item.label}
            >
              <Icon className="w-5 h-5" />
            </Link>
          );
        })}
      </nav>

      {/* User Avatar */}
      <div className="w-10 h-10 bg-gray-700 rounded-full flex items-center justify-center">
        <span className="text-white text-sm font-medium">U</span>
      </div>
    </div>
  );
}
