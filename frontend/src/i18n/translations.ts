import { Language } from '../types';

export const translations: Record<Language, Record<string, string>> = {
  en: {
    // Navigation
    dashboard: 'Overview Dashboard',
    productChecker: 'Product Checker',
    modelComparison: 'Model Comparison',
    networkExplorer: 'Network Explorer',
    clustering: 'Clustering Analysis',
    explainability: 'Explainability (SHAP)',
    experiments: 'Research Experiments',
    reports: 'Reports & Exports',
    settings: 'Settings & Status',

    // App Header & Branding
    appTitle: 'FraudShield ML',
    appSubtitle: 'Fake Review Detection & Network Intelligence',
    researchMode: 'Research Mode',
    demoMode: 'Demo Mode',
    backendOnline: 'API Online',
    backendOffline: 'API Offline',

    // KPI Cards
    totalProducts: 'Total Labeled Products',
    fakePrevalence: 'Fake-Buyer Prevalence',
    activeModels: 'Registered ML Models',
    bestModel: 'Top Validated Model',
    optimalK: 'Optimal Clusters (K-Means)',
    globalNetworkImpact: 'Network Feature Impact',

    // Risk Tiers
    lowRisk: 'Low Risk',
    moderateRisk: 'Moderate Risk',
    highRisk: 'High Risk',
    fakeBuyer: 'Fake-Review Buyer',
    genuineBuyer: 'Genuine Product',

    // Product Checker Form
    catalogLookup: 'Catalog Product Lookup',
    manualInput: 'Manual Feature Entry',
    selectModel: 'Select Inference Model',
    productIdPlaceholder: 'Enter Product ID (e.g., 1001, 1080)...',
    displayedRating: 'Observed Star Rating',
    runPrediction: 'Analyze Product Risk',
    analyzing: 'Evaluating Model...',
    predictionResult: 'Risk Assessment Result',
    riskScore: 'Fraud Risk Score',
    calibratedProbability: 'Calibrated Probability',
    trustAdjustedRating: 'Bayesian Trust-Adjusted Rating',
    multilingualExplanation: 'Multilingual Model Explanation',

    // Common Actions
    exportPdf: 'Export PDF Report',
    exportCsv: 'Export CSV',
    exportJson: 'Export JSON',
    search: 'Search...',
    filterByFeatureGroup: 'Filter by Feature Group',
    filterByEvaluation: 'Filter by Evaluation Protocol',
    all: 'All',
    viewDetails: 'View Details',
    close: 'Close',
    refresh: 'Refresh Data',
    reset: 'Reset',

    // Disclaimer
    disclaimer: 'Academic & Research System: Classifies product-level fake-review buying coordination. Does not guarantee classification of isolated individual reviews.',
  },

  hi: {
    // Navigation
    dashboard: 'डैशबोर्ड अवलोकन',
    productChecker: 'उत्पाद जोखिम जाँच',
    modelComparison: 'मॉडल तुलना',
    networkExplorer: 'नेटवर्क एक्सप्लोरर',
    clustering: 'क्लस्टरिंग विश्लेषण',
    explainability: 'व्याख्यात्मक एआई (SHAP)',
    experiments: 'अनुसंधान प्रयोग',
    reports: 'रिपोर्ट और निर्यात',
    settings: 'सेटिंग्स और स्थिति',

    // App Header & Branding
    appTitle: 'फ्रॉडशील्ड एमएल',
    appSubtitle: 'फेक रिव्यू पहचान एवं नेटवर्क विश्लेषण',
    researchMode: 'रिसर्च मोड',
    demoMode: 'डेमो मोड',
    backendOnline: 'एपीआई सक्रिय',
    backendOffline: 'एपीआई ऑफ़लाइन',

    // KPI Cards
    totalProducts: 'कुल लेबल किए गए उत्पाद',
    fakePrevalence: 'फेक खरीदार प्रसार',
    activeModels: 'पंजीकृत एमएल मॉडल',
    bestModel: 'सर्वश्रेष्ठ मान्य मॉडल',
    optimalK: 'इष्टतम क्लस्टर (K-Means)',
    globalNetworkImpact: 'नेटवर्क फीचर प्रभाव',

    // Risk Tiers
    lowRisk: 'कम जोखिम (Low Risk)',
    moderateRisk: 'मध्यम जोखिम (Moderate Risk)',
    highRisk: 'उच्च जोखिम (High Risk)',
    fakeBuyer: 'फेक-रिव्यू खरीदार',
    genuineBuyer: 'विश्वसनीय उत्पाद',

    // Product Checker Form
    catalogLookup: 'कैटलॉग उत्पाद खोज',
    manualInput: 'मैनुअल फीचर प्रविष्टि',
    selectModel: 'अनुमान मॉडल चुनें',
    productIdPlaceholder: 'उत्पाद आईडी दर्ज करें (उदा. 1001, 1080)...',
    displayedRating: 'दिखाई गई स्टार रेटिंग',
    runPrediction: 'उत्पाद जोखिम का विश्लेषण करें',
    analyzing: 'मॉडल मूल्यांकन जारी है...',
    predictionResult: 'जोखिम मूल्यांकन परिणाम',
    riskScore: 'धोखाधड़ी जोखिम स्कोर',
    calibratedProbability: 'कैलिब्रेटेड संभावना',
    trustAdjustedRating: 'भरोसेमंद संशोधित रेटिंग',
    multilingualExplanation: 'बहुभाषी मॉडल व्याख्या',

    // Common Actions
    exportPdf: 'पीडीएफ रिपोर्ट निर्यात करें',
    exportCsv: 'सीएसवी निर्यात',
    exportJson: 'जेएसओएन निर्यात',
    search: 'खोजें...',
    filterByFeatureGroup: 'फीचर समूह द्वारा फ़िल्टर करें',
    filterByEvaluation: 'मूल्यांकन प्रोटोकॉल द्वारा फ़िल्टर',
    all: 'सभी',
    viewDetails: 'विवरण देखें',
    close: 'बंद करें',
    refresh: 'डेटा ताज़ा करें',
    reset: 'रीसेट',

    // Disclaimer
    disclaimer: 'अकादमिक अनुसंधान प्रणाली: यह उत्पाद स्तर पर संगठित फेक रिव्यू खरीद की पहचान करती है। यह प्रत्येक व्यक्तिगत रिव्यू के फेक होने की गारंटी नहीं देती।',
  },

  hinglish: {
    // Navigation
    dashboard: 'Overview Dashboard',
    productChecker: 'Product Risk Checker',
    modelComparison: 'Model Comparison',
    networkExplorer: 'Network Explorer',
    clustering: 'Clustering Analysis',
    explainability: 'Explainability (SHAP)',
    experiments: 'Research Experiments',
    reports: 'Reports & Exports',
    settings: 'Settings & Status',

    // App Header & Branding
    appTitle: 'FraudShield ML',
    appSubtitle: 'Fake Review Detection & Network Intelligence',
    researchMode: 'Research Mode',
    demoMode: 'Demo Mode',
    backendOnline: 'API Online',
    backendOffline: 'API Disconnected',

    // KPI Cards
    totalProducts: 'Total Labeled Products',
    fakePrevalence: 'Fake-Buyer Prevalence %',
    activeModels: 'Registered ML Models',
    bestModel: 'Best Validated Model',
    optimalK: 'Optimal K-Means Clusters',
    globalNetworkImpact: 'Network Features Ka Impact',

    // Risk Tiers
    lowRisk: 'Low Risk (Surakshit)',
    moderateRisk: 'Moderate Risk (Dhyan De)',
    highRisk: 'High Risk (Sandigdh)',
    fakeBuyer: 'Fake-Review Buyer Product',
    genuineBuyer: 'Genuine Product',

    // Product Checker Form
    catalogLookup: 'Catalog Se Product Dhundhe',
    manualInput: 'Manual Features Daale',
    selectModel: 'Inference Model Chune',
    productIdPlaceholder: 'Product ID enter karein (e.g. 1001, 1080)...',
    displayedRating: 'Displayed Star Rating',
    runPrediction: 'Risk Analyze Karein',
    analyzing: 'Prediction chal rahi hai...',
    predictionResult: 'Risk Assessment Result',
    riskScore: 'Fraud Risk Score (0–100)',
    calibratedProbability: 'Calibrated Probability',
    trustAdjustedRating: 'Trust-Adjusted Rating',
    multilingualExplanation: 'Model Ki Explanation',

    // Common Actions
    exportPdf: 'PDF Report Export Karein',
    exportCsv: 'CSV Export',
    exportJson: 'JSON Export',
    search: 'Search karein...',
    filterByFeatureGroup: 'Feature Group Se Filter Karein',
    filterByEvaluation: 'Split Protocol Se Filter Karein',
    all: 'All',
    viewDetails: 'Details Dekhein',
    close: 'Band Karein',
    refresh: 'Refresh Karein',
    reset: 'Reset',

    // Disclaimer
    disclaimer: 'Academic Research Project: Yeh system coordinated fake review buyers ko detect karta hai, individual reviews ko nahi.',
  },
};

