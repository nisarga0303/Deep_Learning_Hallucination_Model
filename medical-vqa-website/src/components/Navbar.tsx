
import { Stethoscope, Brain, Database, BarChart3, Presentation, Users } from 'lucide-react';

export const Navbar = () => {
  const navItems = [
    { label: 'Overview', href: '#overview', icon: Presentation },
    { label: 'Methodology', href: '#methodology', icon: Brain },
    { label: 'Models & Training', href: '#models', icon: Database },
    { label: 'Results', href: '#results', icon: BarChart3 },
    { label: 'Demo', href: '#demo', icon: Stethoscope },
    { label: 'Team', href: '#team', icon: Users },
  ];

  return (
    <nav className="fixed top-0 left-0 right-0 z-50 glass border-b border-slate-200">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          <div className="flex-shrink-0 flex items-center gap-2">
            <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center">
              <Brain className="w-5 h-5 text-white" />
            </div>
            <span className="font-bold text-xl tracking-tight text-slate-900">MedVQA<span className="text-primary-600">.KG</span></span>
          </div>
          <div className="hidden md:flex space-x-8">
            {navItems.map((item) => (
              <a
                key={item.label}
                href={item.href}
                className="text-sm font-medium text-slate-600 hover:text-primary-600 transition-colors flex items-center gap-1.5"
              >
                <item.icon className="w-4 h-4" />
                {item.label}
              </a>
            ))}
          </div>
        </div>
      </div>
    </nav>
  );
};
