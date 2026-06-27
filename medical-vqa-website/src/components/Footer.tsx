import { Code2, Mail, Building2 } from 'lucide-react';

export const Footer = () => {
  return (
    <footer id="team" className="bg-slate-900 border-t border-slate-800 pt-16 pb-8">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="grid md:grid-cols-3 gap-12 mb-12">
          
          <div className="col-span-1">
            <div className="flex items-center gap-2 mb-4">
              <div className="w-8 h-8 rounded-lg bg-primary-600 flex items-center justify-center">
                <span className="font-bold text-white text-sm">KG</span>
              </div>
              <span className="font-bold text-xl tracking-tight text-white">MedVQA<span className="text-primary-500">.KG</span></span>
            </div>
            <p className="text-slate-400 text-sm leading-relaxed mb-6">
              An academic research project focused on improving the safety and reliability of Multimodal Large Language Models in healthcare through Knowledge Graph grounding.
            </p>
            <div className="flex gap-4">
              <a href="#" className="text-slate-400 hover:text-white transition-colors">
                <Code2 className="w-5 h-5" />
              </a>
              <a href="#" className="text-slate-400 hover:text-white transition-colors">
                <Mail className="w-5 h-5" />
              </a>
            </div>
          </div>

          <div className="col-span-1">
            <h4 className="text-white font-semibold mb-4 flex items-center gap-2">
              <Building2 className="w-4 h-4 text-primary-500" /> Research Lab
            </h4>
            <ul className="space-y-3 text-sm text-slate-400">
              <li>Artificial Intelligence in Medicine Lab</li>
              <li>Department of Computer Science</li>
              <li>Academic University, 2026</li>
            </ul>
          </div>

          <div className="col-span-1">
            <h4 className="text-white font-semibold mb-4">Quick Links</h4>
            <ul className="space-y-2 text-sm">
              <li><a href="#overview" className="text-slate-400 hover:text-primary-400 transition-colors">Overview & Problem</a></li>
              <li><a href="#models" className="text-slate-400 hover:text-primary-400 transition-colors">Model Architectures</a></li>
              <li><a href="#demo" className="text-slate-400 hover:text-primary-400 transition-colors">Interactive Demo</a></li>
              <li><a href="#results" className="text-slate-400 hover:text-primary-400 transition-colors">Evaluation Results</a></li>
            </ul>
          </div>
        </div>

        <div className="border-t border-slate-800 pt-8 flex flex-col md:flex-row justify-between items-center gap-4">
          <p className="text-slate-500 text-sm">
            © {new Date().getFullYear()} MedVQA-KG Research Project. All rights reserved.
          </p>
          <div className="flex gap-6 text-sm text-slate-500">
            <a href="#" className="hover:text-slate-300">Privacy Policy</a>
            <a href="#" className="hover:text-slate-300">Terms of Service</a>
          </div>
        </div>
      </div>
    </footer>
  );
};
