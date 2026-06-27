
import { motion } from 'framer-motion';
import { ArrowRight, BrainCircuit, Activity } from 'lucide-react';

export const Hero = () => {
  return (
    <section id="overview" className="relative pt-32 pb-20 lg:pt-48 lg:pb-32 overflow-hidden">
      {/* Background decoration */}
      <div className="absolute inset-0 -z-10 bg-[radial-gradient(ellipse_at_top_right,_var(--tw-gradient-stops))] from-primary-100/40 via-white to-white" />
      <div className="absolute top-20 right-0 -z-10 w-96 h-96 bg-primary-200/30 rounded-full blur-3xl opacity-50" />
      <div className="absolute bottom-0 left-10 -z-10 w-72 h-72 bg-cyan-100/40 rounded-full blur-3xl opacity-50" />

      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 text-center">
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
        >
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-primary-50 text-primary-700 text-sm font-medium mb-8 border border-primary-100">
            <Activity className="w-4 h-4" />
            <span>Academic AI Research Project</span>
          </div>
          
          <h1 className="text-4xl md:text-6xl font-extrabold text-slate-900 tracking-tight max-w-4xl mx-auto leading-tight mb-6">
            Hallucination Reduction in <span className="text-transparent bg-clip-text bg-gradient-to-r from-primary-600 to-cyan-500">Multimodal LLMs</span> Using Knowledge Graphs
          </h1>
          
          <p className="mt-4 text-xl text-slate-600 max-w-2xl mx-auto mb-10 leading-relaxed">
            A medical visual question answering system that reduces hallucinated answers by grounding predictions in image evidence, text semantics, and medical knowledge graphs.
          </p>

          <div className="flex flex-col sm:flex-row gap-4 justify-center items-center">
            <a href="#models" className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-primary-600 text-white font-medium hover:bg-primary-700 transition-colors shadow-lg shadow-primary-500/30">
              <BrainCircuit className="w-5 h-5" />
              Explore Models
            </a>
            <a href="#demo" className="inline-flex items-center gap-2 px-6 py-3 rounded-lg bg-white text-slate-700 font-medium hover:bg-slate-50 transition-colors border border-slate-200 shadow-sm">
              Try Interactive Demo
              <ArrowRight className="w-4 h-4" />
            </a>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
