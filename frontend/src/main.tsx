import React from 'react';
import ReactDOM from 'react-dom/client';

import { TrafficDashboard } from './components/TrafficDashboard';
import './styles.css';

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <TrafficDashboard />
  </React.StrictMode>,
);
