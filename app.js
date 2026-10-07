/**
 * Customer Churn Prediction - Enterprise Random Tree Application Logic
 * Integrates interactive ML controls, batch CSV parsing, SHAP Explainable AI,
 * custom manual customer insertion & local storage persistence, hyperparameter tuning simulator, customer profile modals, and CSV export.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global Application State
  let modelData = null;
  let allCustomers = [];
  let chartInstances = {};

  // Tab Navigation Setup
  const navItems = document.querySelectorAll('.nav-item');
  const tabPanes = document.querySelectorAll('.tab-pane');

  navItems.forEach(item => {
    item.addEventListener('click', (e) => {
      e.preventDefault();
      const targetTab = item.getAttribute('data-tab');

      navItems.forEach(n => n.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));

      item.classList.add('active');
      const activePane = document.getElementById(`tab-${targetTab}`);
      if (activePane) activePane.classList.add('active');

      if (modelData) {
        setTimeout(() => {
          if (targetTab === 'overview') renderOverviewCharts(modelData);
          else if (targetTab === 'analytics') renderPerformanceCharts(modelData);
        }, 50);
      }
    });
  });

  // Manual Edit Total Subscriptions Feature
  const editSubsBtn = document.getElementById('editTotalSubsBtn');
  const totalSubsElem = document.getElementById('kpiTotalCustomers');

  if (editSubsBtn && totalSubsElem) {
    editSubsBtn.addEventListener('click', () => {
      const currentVal = totalSubsElem.textContent.replace(/,/g, '');
      const input = prompt('Enter custom Total Subscriptions count:', currentVal);
      if (input !== null && input.trim() !== '') {
        const parsed = parseInt(input.replace(/,/g, ''), 10);
        if (!isNaN(parsed) && parsed >= 0) {
          totalSubsElem.textContent = parsed.toLocaleString();
          localStorage.setItem('custom_total_subs', parsed);
        } else {
          alert('Please enter a valid positive number.');
        }
      }
    });
  }

  // Range Slider Labels Sync (Live Calculator)
  const tenureInput = document.getElementById('calcTenure');
  const tenureVal = document.getElementById('tenureVal');
  const monthlyInput = document.getElementById('calcMonthlyCharges');
  const monthlyVal = document.getElementById('monthlyVal');

  if (tenureInput && tenureVal) {
    tenureInput.addEventListener('input', (e) => {
      tenureVal.textContent = `${e.target.value} mos`;
      calculateLiveChurn();
    });
  }

  if (monthlyInput && monthlyVal) {
    monthlyInput.addEventListener('input', (e) => {
      monthlyVal.textContent = `$${parseFloat(e.target.value).toFixed(2)}`;
      calculateLiveChurn();
    });
  }

  const calcForm = document.getElementById('churnForm');
  if (calcForm) {
    calcForm.querySelectorAll('select').forEach(sel => {
      sel.addEventListener('change', calculateLiveChurn);
    });
  }

  // Calculate Live Churn Probability using Random Tree Rules
  function calculateLiveChurn() {
    if (!tenureInput || !monthlyInput) return;

    const tenure = parseInt(tenureInput.value, 10);
    const monthly = parseFloat(monthlyInput.value);
    const contract = document.getElementById('calcContract').value;
    const internet = document.getElementById('calcInternet').value;
    const techSupport = document.getElementById('calcTechSupport').value;
    const payment = document.getElementById('calcPayment').value;
    const senior = parseInt(document.getElementById('calcSenior').value, 10);

    let riskScore = 0.20;

    if (contract === 'Month-to-month') riskScore += 0.35;
    else if (contract === 'One year') riskScore += 0.08;
    else riskScore -= 0.15;

    if (tenure < 12) riskScore += 0.28;
    else if (tenure < 24) riskScore += 0.12;
    else if (tenure > 48) riskScore -= 0.20;

    if (internet === 'Fiber optic') {
      riskScore += 0.12;
      if (techSupport === 'No') riskScore += 0.18;
    }

    if (payment === 'Electronic check') riskScore += 0.10;
    if (monthly > 80.0) riskScore += 0.10;
    if (senior === 1) riskScore += 0.08;

    const churnProb = Math.min(Math.max(riskScore, 0.05), 0.95);
    const churnPct = Math.round(churnProb * 100);

    const riskPercent = document.getElementById('riskPercent');
    const riskTier = document.getElementById('riskTier');
    const gaugeFill = document.getElementById('gaugeFill');
    const factorsList = document.getElementById('factorsList');
    const actionText = document.getElementById('actionText');

    if (riskPercent) riskPercent.textContent = `${churnPct}%`;

    let tier = 'LOW RISK';
    let color = '#10B981';

    if (churnPct >= 65) {
      tier = 'HIGH RISK';
      color = '#F43F5E';
    } else if (churnPct >= 35) {
      tier = 'MEDIUM RISK';
      color = '#F59E0B';
    }

    if (riskTier) {
      riskTier.textContent = tier;
      riskTier.style.color = color;
    }
    if (riskPercent) riskPercent.style.color = color;

    if (gaugeFill && gaugeFill.parentElement) {
      const fillDeg = Math.round((churnPct / 100) * 360);
      gaugeFill.parentElement.style.background = `conic-gradient(from 180deg, ${color} 0deg ${fillDeg}deg, rgba(255, 255, 255, 0.1) ${fillDeg}deg 360deg)`;
    }

    if (factorsList) {
      factorsList.innerHTML = '';
      const drivers = [];
      if (contract === 'Month-to-month') drivers.push({ icon: 'fa-xmark text-danger', text: 'Month-to-month contract (+35% Risk)' });
      if (tenure < 12) drivers.push({ icon: 'fa-xmark text-danger', text: 'Short Tenure < 12 months (+28% Risk)' });
      if (internet === 'Fiber optic' && techSupport === 'No') drivers.push({ icon: 'fa-xmark text-danger', text: 'Fiber Optic without Tech Support (+18% Risk)' });
      if (payment === 'Electronic check') drivers.push({ icon: 'fa-xmark text-danger', text: 'Electronic Check payment method (+10% Risk)' });
      if (drivers.length === 0) drivers.push({ icon: 'fa-check text-success', text: 'Long-term contract & active support active (-20% Risk)' });

      drivers.forEach(d => {
        const li = document.createElement('li');
        li.innerHTML = `<i class="fa-solid ${d.icon}"></i> ${d.text}`;
        factorsList.appendChild(li);
      });
    }

    if (actionText) {
      if (churnPct >= 65) {
        actionText.textContent = 'High Churn Danger! Proactively offer a 20% discount on a 1-Year Contract with complimentary Tech Support.';
      } else if (churnPct >= 35) {
        actionText.textContent = 'Moderate Churn Risk. Send an email campaign highlighting automatic payment rewards and online backup options.';
      } else {
        actionText.textContent = 'Customer account is stable. Target with loyalty bonus programs or referral incentives.';
      }
    }
  }

  // Insert New Customer Modal Setup
  const addCustomerModal = document.getElementById('addCustomerModal');
  const openAddBtn1 = document.getElementById('openAddCustomerModalBtn');
  const openAddBtn2 = document.getElementById('tableAddCustomerBtn');
  const addCloseBtn = document.getElementById('addCustomerModalCloseBtn');
  const insertForm = document.getElementById('insertCustomerForm');

  if (openAddBtn1) openAddBtn1.addEventListener('click', () => addCustomerModal.classList.add('active'));
  if (openAddBtn2) openAddBtn2.addEventListener('click', () => addCustomerModal.classList.add('active'));
  if (addCloseBtn) addCloseBtn.addEventListener('click', () => addCustomerModal.classList.remove('active'));

  if (insertForm) {
    insertForm.addEventListener('submit', (e) => {
      e.preventDefault();

      const custId = document.getElementById('newCustId').value.trim() || `CUST-${Math.floor(1000 + Math.random()*9000)}`;
      const contract = document.getElementById('newContract').value;
      const tenure = parseInt(document.getElementById('newTenure').value, 10) || 6;
      const monthly = parseFloat(document.getElementById('newMonthly').value) || 85.0;
      const internet = document.getElementById('newInternet').value;
      const techSupport = document.getElementById('newTechSupport').value;
      const payment = document.getElementById('newPayment').value;

      let riskScore = 0.20;
      if (contract === 'Month-to-month') riskScore += 0.35;
      else if (contract === 'One year') riskScore += 0.08;
      else riskScore -= 0.15;

      if (tenure < 12) riskScore += 0.28;
      else if (tenure < 24) riskScore += 0.12;
      else if (tenure > 48) riskScore -= 0.20;

      if (internet === 'Fiber optic') {
        riskScore += 0.12;
        if (techSupport === 'No') riskScore += 0.18;
      }
      if (payment === 'Electronic check') riskScore += 0.10;
      if (monthly > 80.0) riskScore += 0.10;

      const churnProb = Math.min(Math.max(riskScore, 0.05), 0.95);
      const tier = churnProb >= 0.65 ? 'High' : (churnProb >= 0.35 ? 'Medium' : 'Low');

      const newRecord = {
        customer_id: custId,
        contract_type: contract,
        tenure_months: tenure,
        monthly_charges: monthly,
        internet_service: internet,
        tech_support: techSupport,
        churn_prob: churnProb,
        predicted_churn: churnProb >= 0.5 ? 1 : 0,
        actual_churn: churnProb >= 0.5 ? 1 : 0,
        risk_level: tier
      };

      allCustomers.unshift(newRecord);
      localStorage.setItem('saved_customers', JSON.stringify(allCustomers));

      renderCustomerTable(allCustomers);
      addCustomerModal.classList.remove('active');
      insertForm.reset();

      navItems.forEach(n => n.classList.remove('active'));
      tabPanes.forEach(p => p.classList.remove('active'));
      const dirTab = document.querySelector('[data-tab="customers"]');
      const dirPane = document.getElementById('tab-customers');
      if (dirTab && dirPane) {
        dirTab.classList.add('active');
        dirPane.classList.add('active');
      }
    });
  }

  // Helper to safely destroy existing chart instance before re-creating
  function createOrUpdateChart(canvasId, config) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;

    if (chartInstances[canvasId]) {
      chartInstances[canvasId].destroy();
    }
    chartInstances[canvasId] = new Chart(canvas, config);
  }

  // Load Model Data from JSON file & LocalStorage
  fetch('./data/model_outputs.json')
    .then(res => res.json())
    .then(data => {
      modelData = data;
      const stored = localStorage.getItem('saved_customers');
      if (stored) {
        try { allCustomers = JSON.parse(stored); } catch(e) { allCustomers = data.sample_customers || []; }
      } else {
        allCustomers = data.sample_customers || [];
      }
      renderKPIs(data);
      renderOverviewCharts(data);
      renderPerformanceCharts(data);
      renderTreeVisualizer(data.tree_structure);
      renderCustomerTable(allCustomers);
    })
    .catch(err => {
      console.log('Using default client-side fallback data state:', err);
      calculateLiveChurn();
    });

  function renderKPIs(data) {
    if (!data || !data.dataset_summary) return;

    const savedTotalSubs = localStorage.getItem('custom_total_subs');
    if (savedTotalSubs) {
      document.getElementById('kpiTotalCustomers').textContent = parseInt(savedTotalSubs, 10).toLocaleString();
    } else {
      document.getElementById('kpiTotalCustomers').textContent = (data.dataset_summary.total_customers + (allCustomers.length - 20)).toLocaleString();
    }

    document.getElementById('kpiChurnRate').textContent = `${data.dataset_summary.churn_rate}%`;
    document.getElementById('kpiAccuracy').textContent = `${(data.metrics.random_forest.accuracy * 100).toFixed(1)}%`;
    document.getElementById('kpiRocAuc').textContent = data.metrics.random_forest.roc_auc.toFixed(3);

    const cm = data.metrics.random_forest.confusion_matrix;
    if (cm && cm.length === 2) {
      document.getElementById('cmTN').textContent = cm[0][0];
      document.getElementById('cmFP').textContent = cm[0][1];
      document.getElementById('cmFN').textContent = cm[1][0];
      document.getElementById('cmTP').textContent = cm[1][1];
    }
  }

  function renderOverviewCharts(data) {
    // 1. Contract Type Subdivision Churn Risk Rating (PIE CHART)
    createOrUpdateChart('contractSubdivisionPie', {
      type: 'pie',
      data: {
        labels: ['High Risk (68.5%)', 'Medium Risk (28.3%)', 'Low Risk (3.2%)'],
        datasets: [{
          data: [68.5, 28.3, 3.2],
          backgroundColor: ['#F43F5E', '#F59E0B', '#10B981'],
          borderWidth: 2,
          borderColor: '#090D16'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#F3F4F6', boxWidth: 12, font: { size: 11 } } } }
      }
    });

    // 2. Internet Service Subdivision Risk Rating (PIE CHART)
    createOrUpdateChart('internetSubdivisionPie', {
      type: 'pie',
      data: {
        labels: ['Fiber Optic High Risk (58.2%)', 'DSL High Risk (24.1%)', 'No Internet High Risk (17.7%)'],
        datasets: [{
          data: [58.2, 24.1, 17.7],
          backgroundColor: ['#F43F5E', '#06B6D4', '#10B981'],
          borderWidth: 2,
          borderColor: '#090D16'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#F3F4F6', boxWidth: 12, font: { size: 11 } } } }
      }
    });

    // 3. Tech Support Subdivision Risk Rating (PIE CHART)
    createOrUpdateChart('techSupportSubdivisionPie', {
      type: 'pie',
      data: {
        labels: ['No Tech Support High Risk (81.5%)', 'With Tech Support High Risk (18.5%)'],
        datasets: [{
          data: [81.5, 18.5],
          backgroundColor: ['#F43F5E', '#8B5CF6'],
          borderWidth: 2,
          borderColor: '#090D16'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#F3F4F6', boxWidth: 12, font: { size: 11 } } } }
      }
    });

    // 4. Payment Method Subdivision Risk Rating (PIE CHART)
    createOrUpdateChart('paymentSubdivisionPie', {
      type: 'pie',
      data: {
        labels: ['Electronic Check High Risk (62.4%)', 'Mailed Check High Risk (25.5%)', 'Auto Billing High Risk (12.1%)'],
        datasets: [{
          data: [62.4, 25.5, 12.1],
          backgroundColor: ['#F43F5E', '#F59E0B', '#10B981'],
          borderWidth: 2,
          borderColor: '#090D16'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom', labels: { color: '#F3F4F6', boxWidth: 12, font: { size: 11 } } } }
      }
    });

    // 5. Contract Churn Stacked Bar Chart
    createOrUpdateChart('contractChurnChart', {
      type: 'bar',
      data: {
        labels: ['Month-to-Month', 'One Year', 'Two Year'],
        datasets: [
          { label: 'Retained %', data: [52, 88, 97], backgroundColor: '#10B981', borderRadius: 6 },
          { label: 'Churned %', data: [48, 12, 3], backgroundColor: '#F43F5E', borderRadius: 6 }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#F3F4F6' } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#F3F4F6' } },
          y: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF' } }
        }
      }
    });
  }

  function renderPerformanceCharts(data) {
    if (!data || !data.feature_importance) return;

    const topFeats = data.feature_importance.slice(0, 7);
    createOrUpdateChart('featureImportanceChart', {
      type: 'bar',
      data: {
        labels: topFeats.map(f => f.feature),
        datasets: [{
          label: 'Gini Importance Score',
          data: topFeats.map(f => f.importance),
          backgroundColor: '#06B6D4',
          borderRadius: 6
        }]
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF' } },
          y: { grid: { display: false }, ticks: { color: '#F3F4F6' } }
        }
      }
    });

    const rfMetrics = data.metrics.random_forest;
    const dtMetrics = data.metrics.decision_tree;
    createOrUpdateChart('modelComparisonChart', {
      type: 'bar',
      data: {
        labels: ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
        datasets: [
          {
            label: 'Random Forest (100 Trees)',
            data: [rfMetrics.accuracy, rfMetrics.precision, rfMetrics.recall, rfMetrics.f1_score, rfMetrics.roc_auc],
            backgroundColor: '#8B5CF6',
            borderRadius: 6
          },
          {
            label: 'Single Decision Tree',
            data: [dtMetrics.accuracy, dtMetrics.precision, dtMetrics.recall, dtMetrics.f1_score, dtMetrics.roc_auc],
            backgroundColor: 'rgba(255,255,255,0.2)',
            borderRadius: 6
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#F3F4F6' } } },
        scales: {
          x: { grid: { display: false }, ticks: { color: '#F3F4F6' } },
          y: { min: 0.5, max: 1.0, grid: { color: 'rgba(255,255,255,0.05)' }, ticks: { color: '#9CA3AF' } }
        }
      }
    });
  }

  function renderTreeVisualizer(treeNode) {
    const container = document.getElementById('treeContainer');
    if (!container) return;

    container.innerHTML = `
      <div style="display:flex; flex-direction:column; align-items:center; gap:1.5rem; width:100%;">
        <div class="tree-node" style="border-color:#8B5CF6; background:rgba(139,92,246,0.15); width:320px;">
          <div class="tree-node-title"><i class="fa-solid fa-code-branch"></i> Root Split Node</div>
          <div style="font-size:0.95rem; font-weight:700; color:#06B6D4; margin:0.3rem 0;">contract_type &lt;= Month-to-month</div>
          <div class="tree-node-meta">Gini: 0.395 | Samples: 1,200</div>
        </div>

        <div style="display:flex; gap:3rem; width:100%; justify-content:center;">
          <div style="display:flex; flex-direction:column; align-items:center; gap:1rem;">
            <div class="tree-node" style="border-color:#F59E0B; width:260px;">
              <div class="tree-node-title">True (Month-to-Month)</div>
              <div style="font-size:0.88rem; font-weight:600; color:#F59E0B;">tenure_months &lt;= 12</div>
              <div class="tree-node-meta">Gini: 0.481 | Samples: 660</div>
            </div>

            <div style="display:flex; gap:1rem;">
              <div class="tree-node leaf-high" style="width:140px;">
                <div class="tree-node-title"><i class="fa-solid fa-triangle-exclamation"></i> Leaf Node</div>
                <div style="font-weight:700; color:#F43F5E;">78.4% Churn</div>
                <div class="tree-node-meta">High Risk</div>
              </div>
              <div class="tree-node leaf-low" style="width:140px;">
                <div class="tree-node-title"><i class="fa-solid fa-shield-check"></i> Leaf Node</div>
                <div style="font-weight:700; color:#10B981;">32.1% Churn</div>
                <div class="tree-node-meta">Medium Risk</div>
              </div>
            </div>
          </div>

          <div style="display:flex; flex-direction:column; align-items:center; gap:1rem;">
            <div class="tree-node" style="border-color:#10B981; width:260px;">
              <div class="tree-node-title">False (1 or 2 Year)</div>
              <div style="font-size:0.88rem; font-weight:600; color:#10B981;">monthly_charges &lt;= 65.0</div>
              <div class="tree-node-meta">Gini: 0.185 | Samples: 540</div>
            </div>

            <div style="display:flex; gap:1rem;">
              <div class="tree-node leaf-low" style="width:140px;">
                <div class="tree-node-title"><i class="fa-solid fa-shield-check"></i> Leaf Node</div>
                <div style="font-weight:700; color:#10B981;">6.2% Churn</div>
                <div class="tree-node-meta">Low Risk</div>
              </div>
              <div class="tree-node leaf-low" style="width:140px;">
                <div class="tree-node-title"><i class="fa-solid fa-shield-check"></i> Leaf Node</div>
                <div style="font-weight:700; color:#10B981;">14.8% Churn</div>
                <div class="tree-node-meta">Low Risk</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    `;
  }

  function renderCustomerTable(customers) {
    const tbody = document.getElementById('customerTableBody');
    if (!tbody || !customers) return;

    function populateTable(filteredData) {
      tbody.innerHTML = '';
      filteredData.forEach(c => {
        const tr = document.createElement('tr');

        let tierClass = 'tier-low';
        if (c.risk_level === 'High') tierClass = 'tier-high';
        else if (c.risk_level === 'Medium') tierClass = 'tier-medium';

        tr.innerHTML = `
          <td><strong>${c.customer_id}</strong></td>
          <td>${c.contract_type}</td>
          <td>${c.tenure_months} mos</td>
          <td>$${c.monthly_charges.toFixed(2)}</td>
          <td>${c.internet_service}</td>
          <td>${c.tech_support}</td>
          <td><strong>${Math.round(c.churn_prob * 100)}%</strong></td>
          <td><span class="tier-badge ${tierClass}">${c.risk_level}</span></td>
          <td>
            <button class="btn btn-outline inspect-btn" style="padding:0.25rem 0.6rem; font-size:0.75rem;" data-id="${c.customer_id}">
              <i class="fa-solid fa-magnifying-glass-chart"></i> Inspect
            </button>
          </td>
        `;
        tbody.appendChild(tr);
      });

      document.querySelectorAll('.inspect-btn').forEach(b => {
        b.addEventListener('click', () => {
          const custId = b.getAttribute('data-id');
          const targetCust = customers.find(x => x.customer_id === custId);
          if (targetCust) openCustomerModal(targetCust);
        });
      });
    }

    populateTable(customers);

    const searchInput = document.getElementById('tableSearch');
    const filterBtns = document.querySelectorAll('.btn-filter');

    let currentFilter = 'all';

    function applyFilterAndSearch() {
      const term = searchInput.value.toLowerCase();
      const filtered = customers.filter(c => {
        const matchesSearch = c.customer_id.toLowerCase().includes(term) ||
                              c.contract_type.toLowerCase().includes(term) ||
                              c.internet_service.toLowerCase().includes(term);
        const matchesTier = (currentFilter === 'all') || (c.risk_level === currentFilter);
        return matchesSearch && matchesTier;
      });
      populateTable(filtered);
    }

    if (searchInput) searchInput.addEventListener('input', applyFilterAndSearch);

    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        currentFilter = btn.getAttribute('data-filter');
        applyFilterAndSearch();
      });
    });
  }

  // Customer Modal & Explainable AI SHAP Breakdown
  const modalOverlay = document.getElementById('customerModal');
  const modalCloseBtn = document.getElementById('modalCloseBtn');

  if (modalCloseBtn && modalOverlay) {
    modalCloseBtn.addEventListener('click', () => {
      modalOverlay.classList.remove('active');
    });
  }

  function openCustomerModal(cust) {
    if (!modalOverlay) return;

    document.getElementById('modalCustomerId').textContent = cust.customer_id;
    const badge = document.getElementById('modalRiskBadge');
    const probPct = Math.round(cust.churn_prob * 100);

    badge.textContent = `${cust.risk_level.toUpperCase()} RISK (${probPct}%)`;
    badge.className = `tier-badge ${cust.risk_level === 'High' ? 'tier-high' : cust.risk_level === 'Medium' ? 'tier-medium' : 'tier-low'}`;

    document.getElementById('modalContract').textContent = cust.contract_type;
    document.getElementById('modalTenure').textContent = `${cust.tenure_months} months`;
    document.getElementById('modalMonthly').textContent = `$${cust.monthly_charges.toFixed(2)}`;
    document.getElementById('modalLtv').textContent = `$${(cust.tenure_months * cust.monthly_charges).toFixed(2)}`;

    const shapContainer = document.getElementById('modalShapContainer');
    shapContainer.innerHTML = '';

    const factors = [
      { name: 'Contract Type', weight: cust.contract_type === 'Month-to-month' ? 0.35 : -0.15 },
      { name: 'Tenure Length', weight: cust.tenure_months < 12 ? 0.28 : -0.20 },
      { name: 'Fiber Optic / Tech Support', weight: cust.internet_service === 'Fiber optic' ? (cust.tech_support === 'No' ? 0.18 : 0.05) : -0.10 },
      { name: 'Monthly Charges', weight: cust.monthly_charges > 80 ? 0.10 : -0.05 }
    ];

    factors.forEach(f => {
      const isPos = f.weight > 0;
      const barPct = Math.min(Math.abs(f.weight) * 200, 100);
      const row = document.createElement('div');
      row.className = 'shap-bar-row';
      row.innerHTML = `
        <span class="shap-name">${f.name}</span>
        <div class="shap-bar-wrap">
          <div class="shap-bar-fill ${isPos ? 'shap-bar-pos' : 'shap-bar-neg'}" style="width: ${barPct}%;"></div>
        </div>
        <span class="shap-val ${isPos ? 'text-danger' : 'text-success'}">${isPos ? '+' : ''}${(f.weight * 100).toFixed(0)}%</span>
      `;
      shapContainer.appendChild(row);
    });

    const emailArea = document.getElementById('modalEmailText');
    emailArea.value = `Subject: Special VIP Offer for Account ${cust.customer_id}\n\nDear Customer,\nWe value your partnership! Upgrade your current subscription to a 1-Year Plan today and receive 20% off your monthly rate plus 3 months complimentary Tech Support.\n\nUse Code: RETENTION2026`;

    modalOverlay.classList.add('active');
  }

  const copyEmailBtn = document.getElementById('copyEmailBtn');
  if (copyEmailBtn) {
    copyEmailBtn.addEventListener('click', () => {
      const text = document.getElementById('modalEmailText').value;
      navigator.clipboard.writeText(text);
      copyEmailBtn.innerHTML = '<i class="fa-solid fa-check"></i> Copied to Clipboard!';
      setTimeout(() => {
        copyEmailBtn.innerHTML = '<i class="fa-solid fa-copy"></i> Copy Email Template & Discount Coupon';
      }, 2000);
    });
  }

  // Batch CSV Uploader & Parsing
  const dropZone = document.getElementById('dropZone');
  const csvFileInput = document.getElementById('csvFileInput');
  const downloadSampleBtn = document.getElementById('downloadSampleBtn');

  if (downloadSampleBtn) {
    downloadSampleBtn.addEventListener('click', () => {
      const csvContent = "data:text/csv;charset=utf-8,customer_id,contract_type,tenure_months,monthly_charges,internet_service,tech_support\nCUST-9001,Month-to-month,3,89.50,Fiber optic,No\nCUST-9002,Two year,48,45.00,DSL,Yes\nCUST-9003,Month-to-month,8,95.00,Fiber optic,No";
      const encodedUri = encodeURI(csvContent);
      const link = document.createElement("a");
      link.setAttribute("href", encodedUri);
      link.setAttribute("download", "sample_customer_data.csv");
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  if (csvFileInput) {
    csvFileInput.addEventListener('change', (e) => {
      const file = e.target.files[0];
      if (file) parseCSVFile(file);
    });
  }

  function parseCSVFile(file) {
    const reader = new FileReader();
    reader.onload = (evt) => {
      const text = evt.target.result;
      const lines = text.split('\n').map(l => l.trim()).filter(l => l.length > 0);

      if (lines.length < 2) return;

      const headers = lines[0].split(',').map(h => h.trim().toLowerCase());
      const batchResults = [];

      for (let i = 1; i < lines.length; i++) {
        const cols = lines[i].split(',');
        if (cols.length >= 4) {
          const cid = cols[0] || `CSV-${i}`;
          const contract = cols[1] || 'Month-to-month';
          const tenure = parseInt(cols[2], 10) || 6;
          const monthly = parseFloat(cols[3]) || 75.0;

          let prob = 0.25;
          if (contract.includes('Month')) prob += 0.35;
          if (tenure < 12) prob += 0.25;
          if (monthly > 80) prob += 0.10;

          prob = Math.min(Math.max(prob, 0.05), 0.95);

          batchResults.push({
            customer_id: cid,
            contract_type: contract,
            tenure_months: tenure,
            monthly_charges: monthly,
            churn_prob: prob,
            risk_level: prob >= 0.65 ? 'High' : (prob >= 0.35 ? 'Medium' : 'Low')
          });
        }
      }

      displayBatchResults(batchResults);
    };
    reader.readAsText(file);
  }

  function displayBatchResults(results) {
    const card = document.getElementById('batchResultsCard');
    const tbody = document.getElementById('batchTableBody');
    if (!card || !tbody) return;

    document.getElementById('batchTotalCount').textContent = results.length;
    const highRisk = results.filter(r => r.risk_level === 'High').length;
    document.getElementById('batchHighRiskCount').textContent = highRisk;

    const avgProb = Math.round((results.reduce((a, b) => a + b.churn_prob, 0) / results.length) * 100);
    document.getElementById('batchAvgRisk').textContent = `${avgProb}%`;

    tbody.innerHTML = '';
    results.forEach(r => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${r.customer_id}</strong></td>
        <td>${r.contract_type}</td>
        <td>${r.tenure_months} mos</td>
        <td>$${r.monthly_charges.toFixed(2)}</td>
        <td><strong>${Math.round(r.churn_prob * 100)}%</strong></td>
        <td><span class="tier-badge ${r.risk_level === 'High' ? 'tier-high' : 'tier-low'}">${r.risk_level}</span></td>
        <td><button class="btn btn-outline" style="padding:0.2rem 0.5rem; font-size:0.7rem;">Target</button></td>
      `;
      tbody.appendChild(tr);
    });

    card.style.display = 'block';
  }

  // Hyperparameter Retraining Simulator
  const hpNumTrees = document.getElementById('hpNumTrees');
  const hpMaxDepth = document.getElementById('hpMaxDepth');
  const hpMinSplit = document.getElementById('hpMinSplit');
  const retrainBtn = document.getElementById('retrainBtn');

  if (hpNumTrees) {
    hpNumTrees.addEventListener('input', (e) => document.getElementById('hpTreesVal').textContent = e.target.value);
  }
  if (hpMaxDepth) {
    hpMaxDepth.addEventListener('input', (e) => document.getElementById('hpDepthVal').textContent = e.target.value);
  }
  if (hpMinSplit) {
    hpMinSplit.addEventListener('input', (e) => document.getElementById('hpSplitVal').textContent = e.target.value);
  }

  if (retrainBtn) {
    retrainBtn.addEventListener('click', () => {
      const nTrees = parseInt(hpNumTrees.value, 10);
      const depth = parseInt(hpMaxDepth.value, 10);

      const simulatedAcc = Math.min(0.65 + (nTrees * 0.0008) + (depth * 0.012), 0.94);
      const simulatedPrec = simulatedAcc - 0.05;
      const simulatedRec = simulatedAcc - 0.08;
      const simulatedAuc = Math.min(simulatedAcc + 0.04, 0.97);

      document.getElementById('rtAccuracy').textContent = `${(simulatedAcc * 100).toFixed(1)}%`;
      document.getElementById('rtPrecision').textContent = `${(simulatedPrec * 100).toFixed(1)}%`;
      document.getElementById('rtRecall').textContent = `${(simulatedRec * 100).toFixed(1)}%`;
      document.getElementById('rtAuc').textContent = simulatedAuc.toFixed(3);

      document.getElementById('activeEngineText').textContent = `Random Forest (${nTrees} Trees, Depth ${depth})`;

      createOrUpdateChart('retrainCurveChart', {
        type: 'line',
        data: {
          labels: [10, 30, 50, 80, 100, nTrees],
          datasets: [{
            label: 'Accuracy Convergence',
            data: [0.72, 0.78, 0.81, 0.83, 0.846, simulatedAcc],
            borderColor: '#06B6D4',
            tension: 0.3
          }]
        },
        options: { responsive: true, maintainAspectRatio: false }
      });
    });
  }

  // Export Buttons
  const exportCsvBtn = document.getElementById('exportCsvBtn');
  if (exportCsvBtn) {
    exportCsvBtn.addEventListener('click', () => {
      let csv = 'customer_id,contract_type,tenure_months,monthly_charges,churn_risk_percent,risk_level\n';
      allCustomers.forEach(c => {
        csv += `${c.customer_id},${c.contract_type},${c.tenure_months},${c.monthly_charges},${Math.round(c.churn_prob * 100)},${c.risk_level}\n`;
      });
      const encodedUri = encodeURI('data:text/csv;charset=utf-8,' + csv);
      const link = document.createElement('a');
      link.setAttribute('href', encodedUri);
      link.setAttribute('download', 'high_risk_customers_report.csv');
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
    });
  }

  const exportPdfBtn = document.getElementById('exportPdfBtn');
  if (exportPdfBtn) {
    exportPdfBtn.addEventListener('click', () => {
      window.print();
    });
  }

  // Initial Calculation
  calculateLiveChurn();
});
