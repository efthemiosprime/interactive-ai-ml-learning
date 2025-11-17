import React from 'react';
import MLUseCasesPanel from '../shared/MLUseCasesPanel';

export default function ProbabilityStatisticsEducationalPanels({ selectedTopic, dataSet }) {
  return (
    <div className="bg-white rounded-xl shadow-lg p-6">
      <h2 className="text-2xl font-bold text-gray-900 mb-6">Educational Content</h2>

      {selectedTopic === 'descriptive' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Descriptive Statistics</h3>
            <p className="text-gray-700 mb-4">
              Descriptive statistics summarize and describe the main features of a dataset. 
              They're essential for understanding data before applying ML algorithms.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Measures:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Mean:</strong> Average value, sensitive to outliers</li>
              <li><strong>Median:</strong> Middle value, robust to outliers</li>
              <li><strong>Variance:</strong> Measures spread around the mean</li>
              <li><strong>Standard Deviation:</strong> Square root of variance, same units as data</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Descriptive Statistics?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Descriptive statistics provide <strong className="text-blue-700">essential insights</strong> about our data 
                before building ML models. They help us understand data quality, detect issues, and make informed decisions 
                about preprocessing and model selection.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔍</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Data Understanding & Quality Check</h4>
                    <p className="text-gray-700">
                      Descriptive statistics reveal <strong>data quality issues</strong> before training. 
                      Mean and standard deviation show if data is centered and scaled properly. 
                      Outliers and anomalies become visible through statistical measures.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> If mean = 1000 and std = 0.1, data has almost no variance - 
                      this feature won't help the model. If mean = 1000 and std = 10000, there are likely outliers.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Feature Preprocessing & Normalization</h4>
                    <p className="text-gray-700">
                      Mean and standard deviation are <strong>essential for feature scaling</strong>. 
                      Standardization (z-score) uses these statistics to normalize features, 
                      ensuring all features contribute equally to the model.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> Age (0-100) and Income (0-1,000,000) have different scales. 
                      Using mean and std, we standardize both to mean=0, std=1, so they're comparable.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🚨</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Outlier Detection & Data Cleaning</h4>
                    <p className="text-gray-700">
                      Standard deviation helps identify <strong>outliers and anomalies</strong>. 
                      Data points beyond 3 standard deviations are likely outliers that can 
                      skew model training and should be handled appropriately.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> If mean income = $50K and std = $10K, an income of $100K 
                      is 5 standard deviations away - likely an outlier that needs investigation.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-100 to-cyan-100 rounded-lg p-4 border-2 border-blue-300">
                <h4 className="font-bold text-blue-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Descriptive statistics are the <strong>first step in any ML pipeline</strong>. 
                  They tell us if our data is ready for modeling, what preprocessing is needed, 
                  and whether we have quality data. Without understanding descriptive statistics, 
                  we're building models blindly - which leads to poor performance and unreliable results.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="descriptive" />
        </div>
      )}

      {selectedTopic === 'covariance' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Covariance & Correlation</h3>
            <p className="text-gray-700 mb-4">
              Covariance measures how two variables change together. Correlation is the normalized 
              version, ranging from -1 (perfect negative) to +1 (perfect positive).
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Positive Covariance:</strong> Variables increase together</li>
              <li><strong>Negative Covariance:</strong> One increases as the other decreases</li>
              <li><strong>Zero Covariance:</strong> Variables are independent</li>
              <li><strong>Correlation:</strong> Normalized covariance (-1 to 1)</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Covariance & Correlation?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Covariance and correlation reveal <strong className="text-blue-700">relationships between features</strong>, 
                which is crucial for feature selection, dimensionality reduction, and understanding how variables interact in ML models.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Feature Selection & Multicollinearity Detection</h4>
                    <p className="text-gray-700">
                      High correlation between features indicates <strong>redundancy</strong>. 
                      If two features are highly correlated, we can remove one without losing information. 
                      This reduces model complexity and prevents multicollinearity issues.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> If "height in cm" and "height in inches" have correlation = 1.0, 
                      we only need one - removing redundancy improves model efficiency.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📉</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Dimensionality Reduction (PCA)</h4>
                    <p className="text-gray-700">
                      The covariance matrix is the <strong>foundation of Principal Component Analysis (PCA)</strong>. 
                      PCA finds directions of maximum variance (principal components) by analyzing the covariance matrix, 
                      enabling effective dimensionality reduction.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> PCA uses covariance matrix to find the direction where data varies most, 
                      reducing 1000 features to 50 principal components while preserving most information.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔗</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Understanding Feature Relationships</h4>
                    <p className="text-gray-700">
                      Correlation helps us understand <strong>how features relate to each other</strong>. 
                      This knowledge guides feature engineering - we can create interaction features 
                      from correlated variables or identify which features work together.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> If "age" and "years of experience" are highly correlated, 
                      we might create a feature like "age × experience" or use only one of them.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-100 to-cyan-100 rounded-lg p-4 border-2 border-blue-300">
                <h4 className="font-bold text-blue-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Covariance and correlation are <strong>essential for understanding feature relationships</strong>. 
                  They help us build better models by removing redundancy, reducing dimensions, and understanding 
                  how variables interact. Without this knowledge, we risk including redundant features that 
                  waste computation and can cause numerical instability.
                </p>
              </div>
            </div>
          </div>

          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="covariance" />
        </div>
      )}

      {selectedTopic === 'conditional-probability' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Conditional Probability</h3>
            <p className="text-gray-700 mb-4">
              Conditional probability P(A|B) represents the probability of event A occurring 
              given that event B has occurred. It's fundamental to probabilistic models.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Key Concepts:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700">
              <li><strong>Dependence:</strong> Events are dependent if P(A|B) ≠ P(A)</li>
              <li><strong>Independence:</strong> Events are independent if P(A|B) = P(A)</li>
              <li><strong>Chain Rule:</strong> P(A∩B) = P(A|B) × P(B)</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Conditional Probability?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Conditional probability captures <strong className="text-blue-700">dependencies between events</strong>, 
                which is essential for building accurate probabilistic models and understanding how information 
                about one variable affects our beliefs about another.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🌳</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Decision Tree Splitting</h4>
                    <p className="text-gray-700">
                      Decision trees split nodes based on <strong>conditional probabilities</strong>. 
                      At each split, we calculate P(class|feature) to determine which feature 
                      best separates the classes, leading to optimal tree construction.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> If P(spam|contains "free") = 0.9 and P(spam|no "free") = 0.1, 
                      splitting on "contains free" effectively separates spam from non-spam.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🕸️</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Bayesian Networks & Probabilistic Models</h4>
                    <p className="text-gray-700">
                      Bayesian networks represent <strong>conditional dependencies</strong> between variables. 
                      Each node's probability depends on its parents: P(node|parents). 
                      This allows modeling complex probabilistic relationships.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> In a medical diagnosis network, P(disease|symptoms, age) 
                      models how disease probability depends on symptoms and age together.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔗</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Modeling Dependencies & Context</h4>
                    <p className="text-gray-700">
                      Conditional probability captures <strong>how context affects probabilities</strong>. 
                      The probability of an event changes when we have additional information, 
                      which is crucial for accurate predictions in real-world scenarios.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> P(rain|cloudy) &gt; P(rain) - knowing it's cloudy 
                      increases rain probability. This context-dependent reasoning improves predictions.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-100 to-cyan-100 rounded-lg p-4 border-2 border-blue-300">
                <h4 className="font-bold text-blue-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Conditional probability is the <strong>mathematical foundation for reasoning under uncertainty</strong>. 
                  It allows us to update our beliefs based on evidence and model how variables depend on each other. 
                  Without conditional probability, we couldn't build probabilistic models, decision trees, 
                  or Bayesian networks - all essential tools in modern ML.
                </p>
              </div>
            </div>
          </div>

          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="conditional-probability" />
        </div>
      )}

      {selectedTopic === 'bayes' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Bayes' Theorem</h3>
            <p className="text-gray-700 mb-4">
              Bayes' theorem provides a way to update probabilities based on new evidence. 
              It's the foundation of Bayesian inference and many ML algorithms.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Components:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Prior:</strong> Initial belief about probability</li>
              <li><strong>Likelihood:</strong> Probability of evidence given hypothesis</li>
              <li><strong>Posterior:</strong> Updated probability after seeing evidence</li>
              <li><strong>Evidence:</strong> Normalizing constant</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Bayes' Theorem?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Bayes' theorem provides a <strong className="text-blue-700">systematic way to update beliefs</strong> 
                with new evidence, which is fundamental to probabilistic reasoning, classification, and learning from data.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📧</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Probabilistic Classification</h4>
                    <p className="text-gray-700">
                      Bayes' theorem enables <strong>probabilistic classification</strong> by computing 
                      P(class|features). We can predict not just the class, but also the probability 
                      of each class, providing uncertainty estimates.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> Naive Bayes computes P(spam|email) = P(email|spam) × P(spam) / P(email), 
                      giving us both the prediction and confidence level.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🔄</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Learning from Evidence</h4>
                    <p className="text-gray-700">
                      Bayes' theorem allows us to <strong>update beliefs incrementally</strong> as we see more data. 
                      Each new observation updates our posterior, which becomes the prior for the next update. 
                      This enables continuous learning and adaptation.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> Start with prior P(disease) = 0.01. After positive test (likelihood = 0.95), 
                      posterior P(disease|test+) = 0.16. With more evidence, beliefs converge to true probability.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎯</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Incorporating Prior Knowledge</h4>
                    <p className="text-gray-700">
                      Bayes' theorem allows us to <strong>combine prior knowledge with data</strong>. 
                      When we have domain expertise or historical information, we can encode it as priors, 
                      making models more accurate with less data.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> In medical diagnosis, we know disease prevalence (prior). 
                      Bayes' theorem combines this with test results (likelihood) to get accurate diagnosis (posterior).
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-100 to-cyan-100 rounded-lg p-4 border-2 border-blue-300">
                <h4 className="font-bold text-blue-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Bayes' theorem is the <strong>mathematical framework for learning from evidence</strong>. 
                  It transforms the problem of "what should I believe?" into a systematic calculation that 
                  combines prior knowledge with new observations. Without Bayes' theorem, we couldn't build 
                  probabilistic classifiers, update beliefs with data, or reason about uncertainty - 
                  all essential capabilities in modern ML.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="bayes" />
        </div>
      )}

      {selectedTopic === 'distributions' && (
        <div className="space-y-6">
          <div>
            <h3 className="text-xl font-semibold text-blue-900 mb-3">Probability Distributions</h3>
            <p className="text-gray-700 mb-4">
              Probability distributions describe how probabilities are distributed over possible outcomes. 
              Different distributions model different types of data and uncertainty.
            </p>
            <h4 className="font-semibold text-gray-800 mb-2">Common Distributions:</h4>
            <ul className="list-disc list-inside space-y-2 text-gray-700 mb-4">
              <li><strong>Normal:</strong> Continuous, symmetric, many natural phenomena</li>
              <li><strong>Bernoulli:</strong> Binary outcomes (0 or 1)</li>
              <li><strong>Binomial:</strong> Number of successes in n trials</li>
              <li><strong>Poisson:</strong> Count of events in fixed interval</li>
            </ul>
          </div>

          {/* Why Do We Need Them? */}
          <div className="bg-gradient-to-r from-yellow-50 to-orange-50 rounded-lg p-6 border-2 border-yellow-300">
            <h3 className="text-xl font-bold text-yellow-900 mb-4 flex items-center gap-2">
              <span className="text-2xl">🎯</span>
              Why Do We Need Probability Distributions?
            </h3>
            <div className="space-y-4">
              <p className="text-gray-800 text-lg leading-relaxed">
                Probability distributions provide <strong className="text-blue-700">mathematical models</strong> 
                for uncertainty and data generation processes. They guide loss function selection, enable generative 
                modeling, and help us understand and model real-world phenomena.
              </p>
              
              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📉</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Loss Function Design</h4>
                    <p className="text-gray-700">
                      Different distributions lead to <strong>different loss functions</strong>. 
                      Assuming errors follow a normal distribution leads to Mean Squared Error (MSE). 
                      Assuming Bernoulli distribution leads to Cross-Entropy loss. 
                      Choosing the right distribution ensures optimal model training.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> Regression assumes normal errors → MSE loss. 
                      Classification assumes Bernoulli → Cross-entropy loss. 
                      Wrong assumption = wrong loss = poor model performance.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">🎨</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Generative Modeling</h4>
                    <p className="text-gray-700">
                      Distributions enable <strong>generative models</strong> that can create new data. 
                      By learning the distribution P(x) from training data, we can sample new examples 
                      that follow the same distribution, enabling data generation and augmentation.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> GANs learn image distribution P(image), then sample to generate 
                      new realistic images. VAEs learn latent distributions to generate diverse outputs.
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-white rounded-lg p-4 border-2 border-yellow-200">
                <div className="flex items-start gap-3 mb-3">
                  <div className="text-2xl">📊</div>
                  <div>
                    <h4 className="font-bold text-blue-900 mb-2">Uncertainty Quantification</h4>
                    <p className="text-gray-700">
                      Distributions provide <strong>uncertainty estimates</strong> for predictions. 
                      Instead of just predicting a value, we can predict a distribution, 
                      giving us confidence intervals and uncertainty quantification.
                    </p>
                    <div className="mt-2 bg-blue-50 rounded p-2 text-sm text-blue-800">
                      <strong>Example:</strong> Instead of predicting "house price = $300K", 
                      predict "price ~ Normal(μ=$300K, σ=$20K)", giving us uncertainty: 
                      "likely between $280K-$320K with 68% confidence".
                    </div>
                  </div>
                </div>
              </div>

              <div className="bg-gradient-to-r from-blue-100 to-cyan-100 rounded-lg p-4 border-2 border-blue-300">
                <h4 className="font-bold text-blue-900 mb-2">💡 Key Insight:</h4>
                <p className="text-gray-800">
                  Probability distributions are the <strong>language of uncertainty</strong> in ML. 
                  They tell us what loss functions to use, enable generative modeling, and provide 
                  uncertainty quantification. Without understanding distributions, we can't properly 
                  model data, design loss functions, or quantify prediction uncertainty - 
                  all critical for building robust ML systems.
                </p>
              </div>
            </div>
          </div>
          
          {/* Real-World ML Use Cases */}
          <MLUseCasesPanel domain="probability-statistics" operationType="distributions" />
        </div>
      )}
    </div>
  );
}

