
import { motion } from 'framer-motion';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, RadarChart, PolarGrid, PolarAngleAxis, PolarRadiusAxis, Radar } from 'recharts';

const performanceData = [
  { metric: 'Accuracy', Traditional: 66.0, Hybrid: 78.3 },
  { metric: 'Precision', Traditional: 64.0, Hybrid: 76.0 },
  { metric: 'Recall', Traditional: 63.0, Hybrid: 74.0 },
  { metric: 'F1-Score', Traditional: 63.5, Hybrid: 75.0 },
];

export const Results = () => {
  return (
    <section id="results" className="py-20 bg-white">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
        <div className="text-center mb-16">
          <h2 className="text-3xl font-bold text-slate-900 sm:text-4xl">Model Performance & Accuracy</h2>
          <p className="mt-4 text-lg text-slate-600 max-w-2xl mx-auto">
            Reported metrics reflect overall VQA performance rather than only organ and modality classification, giving a more realistic view of end-to-end question answering quality.
          </p>
        </div>

        <div className="grid lg:grid-cols-2 gap-12">
          {/* Bar Chart */}
          <motion.div 
            initial={{ opacity: 0, x: -20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-slate-50 p-6 rounded-2xl border border-slate-200 shadow-sm"
          >
            <h3 className="text-xl font-bold text-slate-900 mb-6 text-center">Comparative Metrics (%)</h3>
            <div className="h-80">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart
                  data={performanceData}
                  margin={{ top: 20, right: 30, left: 20, bottom: 5 }}
                >
                  <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#e2e8f0" />
                  <XAxis dataKey="metric" axisLine={false} tickLine={false} tick={{ fill: '#64748b' }} />
                  <YAxis axisLine={false} tickLine={false} tick={{ fill: '#64748b' }} domain={[0, 100]} />
                  <Tooltip 
                    cursor={{ fill: '#f1f5f9' }}
                    contentStyle={{ borderRadius: '0.5rem', border: 'none', boxShadow: '0 4px 6px -1px rgba(0, 0, 0, 0.1)' }}
                  />
                  <Legend wrapperStyle={{ paddingTop: '20px' }} />
                  <Bar dataKey="Traditional" fill="#94a3b8" radius={[4, 4, 0, 0]} name="Traditional Baseline" />
                  <Bar dataKey="Hybrid" fill="#3b82f6" radius={[4, 4, 0, 0]} name="Hybrid KG-Fusion (Ours)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>

          {/* Radar Chart */}
          <motion.div 
            initial={{ opacity: 0, x: 20 }}
            whileInView={{ opacity: 1, x: 0 }}
            viewport={{ once: true }}
            className="bg-slate-50 p-6 rounded-2xl border border-slate-200 shadow-sm flex flex-col"
          >
            <h3 className="text-xl font-bold text-slate-900 mb-2 text-center">Performance Radar</h3>
            <p className="text-sm text-slate-500 text-center mb-6">Multi-dimensional evaluation analysis</p>
            <div className="h-72 flex-grow">
              <ResponsiveContainer width="100%" height="100%">
                <RadarChart cx="50%" cy="50%" outerRadius="70%" data={performanceData}>
                  <PolarGrid stroke="#cbd5e1" />
                  <PolarAngleAxis dataKey="metric" tick={{ fill: '#475569', fontSize: 14, fontWeight: 500 }} />
                  <PolarRadiusAxis angle={30} domain={[0, 100]} tick={{ fill: '#94a3b8' }} />
                  <Radar name="Traditional" dataKey="Traditional" stroke="#94a3b8" fill="#94a3b8" fillOpacity={0.3} />
                  <Radar name="Hybrid" dataKey="Hybrid" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.5} />
                  <Legend />
                  <Tooltip />
                </RadarChart>
              </ResponsiveContainer>
            </div>
          </motion.div>
        </div>

        <motion.div 
          initial={{ opacity: 0, y: 20 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true }}
          className="mt-12 bg-primary-50 rounded-2xl p-8 border border-primary-100"
        >
          <h3 className="text-xl font-bold text-slate-900 mb-4">Key Findings</h3>
          <div className="grid md:grid-cols-3 gap-6">
            <div className="bg-white p-5 rounded-xl shadow-sm">
              <div className="text-3xl font-extrabold text-primary-600 mb-2">+9%</div>
              <div className="font-semibold text-slate-800">Overall Accuracy Gain</div>
              <p className="text-sm text-slate-600 mt-1">Including answer accuracy raises the hybrid model from 66.0% to 78.3% overall accuracy.</p>
            </div>
            <div className="bg-white p-5 rounded-xl shadow-sm">
              <div className="text-3xl font-extrabold text-primary-600 mb-2">+12%</div>
              <div className="font-semibold text-slate-800">Precision Lift</div>
              <p className="text-sm text-slate-600 mt-1">The hybrid model maintains stronger precision across mixed medical question types.</p>
            </div>
            <div className="bg-white p-5 rounded-xl shadow-sm">
              <div className="text-3xl font-extrabold text-primary-600 mb-2">+11.5%</div>
              <div className="font-semibold text-slate-800">F1-Score Gain</div>
              <p className="text-sm text-slate-600 mt-1">The hybrid pipeline gives a better balance between precision and recall than the traditional baseline.</p>
            </div>
          </div>
        </motion.div>
      </div>
    </section>
  );
};
