
import { motion } from 'framer-motion';
import { Target, Database, GitMerge, FileQuestion } from 'lucide-react';

export const Overview = () => {
  return (
    <section id="methodology" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        
        {/* Problem Statement */}
        <div className="mb-20">
          <div className="text-center mb-12">
            <h2 className="text-3xl font-bold text-slate-900 sm:text-4xl">The Hallucination Problem</h2>
            <p className="mt-4 text-lg text-slate-600 max-w-3xl mx-auto">
              While Multimodal LLMs excel at medical visual question answering, they frequently generate "hallucinated" answers—plausible-sounding medical statements that are not supported by the actual image evidence or established clinical knowledge.
            </p>
          </div>
          
          <div className="grid md:grid-cols-3 gap-8">
            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
              <div className="w-12 h-12 bg-red-100 text-red-600 rounded-xl flex items-center justify-center mb-4">
                <FileQuestion className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">Visual Disconnect</h3>
              <p className="text-slate-600">Models often ignore subtle visual features in X-Rays or MRIs, relying instead on language priors to guess answers.</p>
            </div>
            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
              <div className="w-12 h-12 bg-amber-100 text-amber-600 rounded-xl flex items-center justify-center mb-4">
                <Target className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">Knowledge Deficit</h3>
              <p className="text-slate-600">Standard models lack explicit structured medical knowledge, leading to biologically or clinically impossible predictions.</p>
            </div>
            <div className="bg-slate-50 p-6 rounded-2xl border border-slate-200">
              <div className="w-12 h-12 bg-green-100 text-green-600 rounded-xl flex items-center justify-center mb-4">
                <GitMerge className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-bold text-slate-900 mb-2">Our Solution</h3>
              <p className="text-slate-600">Fusing retrieved Knowledge Graph (KG) triples directly into the multi-modal attention layers to verify answers before output.</p>
            </div>
          </div>
        </div>

        {/* Dataset */}
        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="bg-primary-900 rounded-3xl overflow-hidden shadow-xl"
        >
          <div className="grid md:grid-cols-2">
            <div className="p-10 md:p-12 flex flex-col justify-center">
              <h3 className="text-sm font-semibold text-primary-400 uppercase tracking-wider mb-2">Evaluation Benchmark</h3>
              <h2 className="text-3xl font-bold text-white mb-6">SLAKE Dataset</h2>
              <p className="text-primary-100 mb-8 leading-relaxed">
                We evaluated our hallucination reduction techniques on SLAKE, a comprehensively annotated medical visual question answering dataset containing bilingual (English & Chinese) QA pairs, dense visual annotations, and semantic segmentation masks.
              </p>
              
              <div className="grid grid-cols-2 gap-6">
                <div>
                  <div className="text-3xl font-bold text-white mb-1">642</div>
                  <div className="text-sm text-primary-300">Medical Image Cases</div>
                </div>
                <div>
                  <div className="text-3xl font-bold text-white mb-1">14,028</div>
                  <div className="text-sm text-primary-300">QA Pairs</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-white mb-1">CT, MRI, X-Ray</div>
                  <div className="text-sm text-primary-300">Image Modalities</div>
                </div>
                <div>
                  <div className="text-xl font-bold text-white mb-1">KG Grounded</div>
                  <div className="text-sm text-primary-300">Question Types</div>
                </div>
              </div>
            </div>
            
            <div className="bg-slate-800 p-8 flex items-center justify-center relative overflow-hidden">
               <div className="absolute inset-0 bg-[url('https://images.unsplash.com/photo-1559757175-5700dde675bc?q=80&w=800&auto=format&fit=crop')] bg-cover bg-center opacity-20 mix-blend-luminosity"></div>
               <div className="relative z-10 w-full max-w-sm">
                 <div className="bg-slate-900/80 backdrop-blur border border-slate-700 rounded-xl p-6 shadow-2xl">
                    <div className="flex items-center gap-3 mb-4 border-b border-slate-700 pb-4">
                      <Database className="w-6 h-6 text-primary-400" />
                      <h4 className="text-lg font-bold text-white">Data Annotation</h4>
                    </div>
                    <ul className="space-y-3 text-slate-300 text-sm">
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-cyan-400" /> Open-ended questions</li>
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-cyan-400" /> Closed-ended (Yes/No) questions</li>
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-cyan-400" /> Bounding box detection targets</li>
                      <li className="flex items-center gap-2"><div className="w-1.5 h-1.5 rounded-full bg-cyan-400" /> Organ segmentation masks</li>
                    </ul>
                 </div>
               </div>
            </div>
          </div>
        </motion.div>

      </div>
    </section>
  );
};
