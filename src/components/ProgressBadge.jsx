import { CheckCircle, Clock, Circle } from 'lucide-react';

export default function ProgressBadge({ status }) {
  const configs = {
    completed: {
      icon: CheckCircle,
      label: 'Completed',
      className: 'bg-green-100 text-green-700 border-green-200',
      iconColor: 'text-green-600',
    },
    'in-progress': {
      icon: Clock,
      label: 'In Progress',
      className: 'bg-yellow-100 text-yellow-700 border-yellow-200',
      iconColor: 'text-yellow-600',
    },
    'not-started': {
      icon: Circle,
      label: 'Not Started',
      className: 'bg-gray-100 text-gray-600 border-gray-200',
      iconColor: 'text-gray-400',
    },
  };

  const config = configs[status] || configs['not-started'];
  const Icon = config.icon;

  return (
    <span
      className={`
        inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full
        text-xs font-medium border
        ${config.className}
      `}
    >
      <Icon className={`w-3.5 h-3.5 ${config.iconColor}`} />
      {config.label}
    </span>
  );
}
