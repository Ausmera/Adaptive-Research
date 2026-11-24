#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <iomanip>
#include <fstream>
#include <string>
#include <numeric>

struct Sample {
    double x1, x2, x3, target;
};

struct Neuron {
    double w1, w2, w3, bias, outW;
};

class Network {
    std::vector<Neuron> neurons;
    double outBias = 0.0;
    std::mt19937 rng;
    std::uniform_real_distribution<double> dist{-1.0, 1.0};
    double lr = 0.3;

    static double sigmoid(double x) {
        // Не даём exp() улететь в бесконечность на больших значениях.
        x = std::clamp(x, -30.0, 30.0);
        return 1.0 / (1.0 + std::exp(-x));
    }

public:
    Network(int count, unsigned seed) : rng(seed) {
        for (int i = 0; i < count; ++i)
            addNeuron(false, false);
    }

    void addNeuron(bool log = false, bool neutral = false) {
        // Нулевой выходной вес нужен, чтобы новый нейрон не ломал текущую модель.
        neurons.push_back({dist(rng), dist(rng), dist(rng), dist(rng),
                           neutral ? 0.0 : dist(rng)});

        if (log)
            std::cout << "[GROW] neurons=" << neurons.size()
                      << (neutral ? " neutral" : "") << '\n';
    }

    size_t size() const {
        return neurons.size();
    }

    double predict(double x1, double x2, double x3) const {
        double sum = outBias;

        for (const auto& n : neurons)
            sum += sigmoid(n.w1 * x1 + n.w2 * x2 + n.w3 * x3 + n.bias) * n.outW;

        return sigmoid(sum);
    }

    void train(const Sample& s) {
        std::vector<double> h;
        h.reserve(neurons.size());
        double sum = outBias;

        for (const auto& n : neurons) {
            h.push_back(sigmoid(n.w1 * s.x1 + n.w2 * s.x2 + n.w3 * s.x3 + n.bias));
            sum += h.back() * n.outW;
        }

        double out = sigmoid(sum);
        double dOut = (s.target - out) * out * (1.0 - out);
        outBias += lr * dOut;

        for (size_t i = 0; i < neurons.size(); ++i) {
            auto& n = neurons[i];

            // Считаем dH до обновления outW, иначе градиент будет уже от новых весов.
            double dH = dOut * n.outW * h[i] * (1.0 - h[i]);
            n.outW += lr * dOut * h[i];
            n.w1 += lr * dH * s.x1;
            n.w2 += lr * dH * s.x2;
            n.w3 += lr * dH * s.x3;
            n.bias += lr * dH;
        }
    }

    double mse(const std::vector<Sample>& data) const {
        if (data.empty()) return 0.0;

        double sum = 0.0;
        for (const auto& s : data) {
            double e = s.target - predict(s.x1, s.x2, s.x3);
            sum += e * e;
        }

        return sum / data.size();
    }
};

enum class Strategy {
    Fixed,
    Baseline,
    Consistency,
    Persistence
};

struct Result {
    std::string architecture;
    int trial = 0;
    unsigned seed = 0;
    bool trainSuccess = false;
    bool validationSuccess = false;

    int mainEpochs = 0;
    int baselineProbeEpochs = 0;
    int candidateProbeEpochs = 0;
    int accepted = 0;
    int rejected = 0;
    int neurons = 0;

    double finalTrainMSE = 0.0;
    double finalValidationMSE = 0.0;
    double lastGain1 = 0.0;
    double lastGain2 = 0.0;
    double lastGain3 = 0.0;

    long long neuronSampleWork = 0;

    int totalEpochs() const {
        return mainEpochs + baselineProbeEpochs + candidateProbeEpochs;
    }
};

void trainEpoch(Network& net, const std::vector<Sample>& data,
                std::vector<int>& order, std::mt19937& rng) {
    std::shuffle(order.begin(), order.end(), rng);
    for (int idx : order)
        net.train(data[static_cast<size_t>(idx)]);
}

void trainMany(Network& net, const std::vector<Sample>& data,
               int epochs, unsigned seed, long long& work) {
    std::mt19937 rng(seed);
    std::vector<int> order(data.size());
    std::iota(order.begin(), order.end(), 0);

    for (int e = 0; e < epochs; ++e) {
        trainEpoch(net, data, order, rng);
        work += static_cast<long long>(data.size()) * static_cast<long long>(net.size());
    }
}

Result runTrial(const std::vector<Sample>& trainData,
                const std::vector<Sample>& validation,
                const std::string& name, int initialNeurons,
                Strategy strategy, int trial, unsigned seed) {
    constexpr int maxBudget = 12000;
    constexpr int interval = 300;
    constexpr int maxNeurons = 16;
    constexpr int probeTotal = 1200;
    constexpr int chunks = 3;

    constexpr double targetTrainMSE = 0.01;
    constexpr double targetValidationMSE = 0.05;
    constexpr double minimumGain = 0.00005;
    constexpr double persistenceRatio = 0.80;

    Network net(initialNeurons, seed);
    Result r;
    r.architecture = name;
    r.trial = trial;
    r.seed = seed;

    std::mt19937 mainRng(seed + 999);
    std::vector<int> order(trainData.size());
    std::iota(order.begin(), order.end(), 0);

    double previousTrain = 1e9;
    int stagnant = 0;

    while (r.totalEpochs() < maxBudget) {
        trainEpoch(net, trainData, order, mainRng);
        ++r.mainEpochs;
        r.neuronSampleWork += static_cast<long long>(trainData.size()) *
                              static_cast<long long>(net.size());

        if (r.mainEpochs % interval != 0)
            continue;

        double currentTrain = net.mse(trainData);
        double currentVal = net.mse(validation);

        if (currentTrain <= targetTrainMSE && currentVal <= targetValidationMSE) {
            r.trainSuccess = true;
            r.validationSuccess = true;
            break;
        }

        if (strategy != Strategy::Fixed) {
            if (previousTrain - currentTrain < 0.0005)
                ++stagnant;
            else
                stagnant = 0;

            const int probeCost = probeTotal * 2;
            if (stagnant >= 2 && net.size() < static_cast<size_t>(maxNeurons) &&
                r.totalEpochs() + probeCost <= maxBudget) {

                Network original = net;
                Network baseline = net;
                Network candidate = net;
                candidate.addNeuron(false, true);

                unsigned pseed = seed + static_cast<unsigned>(r.mainEpochs * 31 + 17);
                std::vector<double> baseVal, candVal;
                int chunk = probeTotal / chunks;

                // Обе ветки получают одинаковый порядок примеров на каждом отрезке.
                for (int cp = 0; cp < chunks; ++cp) {
                    unsigned partSeed = pseed + static_cast<unsigned>(cp);
                    trainMany(baseline, trainData, chunk, partSeed, r.neuronSampleWork);
                    trainMany(candidate, trainData, chunk, partSeed, r.neuronSampleWork);
                    baseVal.push_back(baseline.mse(validation));
                    candVal.push_back(candidate.mse(validation));
                }

                double g1 = baseVal[0] - candVal[0];
                double g2 = baseVal[1] - candVal[1];
                double g3 = baseVal[2] - candVal[2];
                r.lastGain1 = g1;
                r.lastGain2 = g2;
                r.lastGain3 = g3;

                bool accept = false;
                if (strategy == Strategy::Baseline)
                    accept = (g3 >= minimumGain && candVal[2] <= targetValidationMSE);
                else if (strategy == Strategy::Consistency)
                    accept = (g1 >= minimumGain && g2 >= minimumGain &&
                              g3 >= minimumGain && candVal[2] <= targetValidationMSE);
                else if (strategy == Strategy::Persistence)
                    accept = (g2 >= minimumGain && g3 >= minimumGain &&
                              g3 >= persistenceRatio * g2 &&
                              candVal[2] <= targetValidationMSE);

                if (accept) {
                    net = candidate;
                    ++r.accepted;
                } else {
                    net = original;
                    ++r.rejected;
                }

                r.baselineProbeEpochs += probeTotal;
                r.candidateProbeEpochs += probeTotal;
                stagnant = 0;
            }
        }

        previousTrain = currentTrain;
    }

    r.finalTrainMSE = net.mse(trainData);
    r.finalValidationMSE = net.mse(validation);
    r.neurons = static_cast<int>(net.size());
    r.trainSuccess = (r.finalTrainMSE <= targetTrainMSE);
    r.validationSuccess = (r.finalValidationMSE <= targetValidationMSE);

    return r;
}

void report(const std::string& name, const std::vector<Result>& rs) {
    int ts = 0, vs = 0;
    double epochs = 0.0, train = 0.0, val = 0.0;
    double neurons = 0.0, accept = 0.0, reject = 0.0, work = 0.0;

    for (const auto& r : rs) {
        ts += r.trainSuccess;
        vs += r.validationSuccess;
        epochs += r.totalEpochs();
        train += r.finalTrainMSE;
        val += r.finalValidationMSE;
        neurons += r.neurons;
        accept += r.accepted;
        reject += r.rejected;
        work += r.neuronSampleWork;
    }

    double n = rs.size();
    std::cout << "\n===== " << name << " =====\n"
              << "Train success: " << ts << "/" << rs.size() << " (" << 100 * ts / n << "%)\n"
              << "Validation success: " << vs << "/" << rs.size() << " (" << 100 * vs / n << "%)\n"
              << "Avg total epochs: " << epochs / n << "\n"
              << "Avg train MSE: " << train / n << "\n"
              << "Avg validation MSE: " << val / n << "\n"
              << "Avg neurons: " << neurons / n << "\n"
              << "Avg accepted growth: " << accept / n << "\n"
              << "Avg rejected growth: " << reject / n << "\n"
              << "Avg neuron-sample work: " << work / n << '\n';
}

void writeCSV(const std::vector<Result>& all) {
    std::ofstream f("results/E10_trials.csv");
    if (!f) {
        std::cerr << "Cannot write results/E10_trials.csv\n";
        return;
    }

    f << "architecture,trial,seed,train_success,validation_success,main_epochs,"
         "baseline_probe_epochs,candidate_probe_epochs,total_epochs,neuron_sample_work,"
         "train_mse,validation_mse,neurons,accepted_growth,rejected_growth,"
         "gain_checkpoint1,gain_checkpoint2,gain_checkpoint3\n";

    for (const auto& r : all) {
        f << r.architecture << ',' << r.trial << ',' << r.seed << ','
          << r.trainSuccess << ',' << r.validationSuccess << ','
          << r.mainEpochs << ',' << r.baselineProbeEpochs << ','
          << r.candidateProbeEpochs << ',' << r.totalEpochs() << ','
          << r.neuronSampleWork << ',' << r.finalTrainMSE << ','
          << r.finalValidationMSE << ',' << r.neurons << ','
          << r.accepted << ',' << r.rejected << ','
          << r.lastGain1 << ',' << r.lastGain2 << ',' << r.lastGain3 << '\n';
    }

    std::ofstream s("results/E10_summary.csv");
    if (!s) {
        std::cerr << "Cannot write results/E10_summary.csv\n";
        return;
    }

    s << "architecture,trials,train_successes,validation_successes,avg_total_epochs,"
         "avg_neuron_sample_work,avg_train_mse,avg_validation_mse,avg_neurons,"
         "avg_accepted_growth,avg_rejected_growth\n";

    for (const std::string name : {"Fixed2", "Fixed3", "Fixed4", "AdaptiveBaseline",
                                   "AdaptiveConsistency", "AdaptivePersistence"}) {
        int count = 0, ts = 0, vs = 0;
        double ep = 0.0, work = 0.0, tr = 0.0, va = 0.0;
        double ne = 0.0, ac = 0.0, re = 0.0;

        for (const auto& r : all) {
            if (r.architecture != name) continue;
            ++count;
            ts += r.trainSuccess;
            vs += r.validationSuccess;
            ep += r.totalEpochs();
            work += r.neuronSampleWork;
            tr += r.finalTrainMSE;
            va += r.finalValidationMSE;
            ne += r.neurons;
            ac += r.accepted;
            re += r.rejected;
        }

        if (!count) continue;
        double n = count;
        s << name << ',' << count << ',' << ts << ',' << vs << ','
          << ep / n << ',' << work / n << ',' << tr / n << ',' << va / n << ','
          << ne / n << ',' << ac / n << ',' << re / n << '\n';
    }
}

int main() {
    const std::vector<Sample> allData = {
        {0, 0, 0, 0}, {0, 0, 1, 1}, {0, 1, 0, 1}, {0, 1, 1, 0},
        {1, 0, 0, 1}, {1, 0, 1, 0}, {1, 1, 0, 0}, {1, 1, 1, 1}
    };

    constexpr int trials = 30;
    std::vector<Result> all;
    all.reserve(trials * 6);

    std::cout << std::fixed << std::setprecision(6)
              << "========================================\n"
              << " AUSMERA AR-01 / EXPERIMENT 10\n"
              << " CONTROLLED GROWTH VALIDATION\n"
              << "========================================\n"
              << "Trials per condition: " << trials << "\n"
              << "Train/validation: 6/2 samples, validation balanced by class\n"
              << "Validation success threshold: MSE <= 0.05; train target: MSE <= 0.01\n"
              << "Budget: 12000 epoch-equivalents per trial; probe: 1200 epochs per branch\n"
              << "Neuron-sample work: training sample presentations weighted by current neuron count\n"
              << "========================================\n";

    for (int i = 0; i < trials; ++i) {
        unsigned seed = 100u + static_cast<unsigned>(i);
        std::mt19937 splitRng(seed + 2024);
        std::vector<int> zeros, ones;

        for (int j = 0; j < 8; ++j)
            (allData[j].target == 0 ? zeros : ones).push_back(j);

        std::shuffle(zeros.begin(), zeros.end(), splitRng);
        std::shuffle(ones.begin(), ones.end(), splitRng);
        std::vector<int> valIdx = {zeros[0], ones[0]};

        std::vector<Sample> trainData, validation;
        for (int j = 0; j < 8; ++j) {
            if (j == valIdx[0] || j == valIdx[1])
                validation.push_back(allData[j]);
            else
                trainData.push_back(allData[j]);
        }

        std::cout << "\nTrial " << i + 1 << "/" << trials
                  << " | Seed " << seed
                  << " | val indices " << valIdx[0] << ',' << valIdx[1] << '\n';

        all.push_back(runTrial(trainData, validation, "Fixed2", 2, Strategy::Fixed, i + 1, seed));
        all.push_back(runTrial(trainData, validation, "Fixed3", 3, Strategy::Fixed, i + 1, seed));
        all.push_back(runTrial(trainData, validation, "Fixed4", 4, Strategy::Fixed, i + 1, seed));
        all.push_back(runTrial(trainData, validation, "AdaptiveBaseline", 2, Strategy::Baseline, i + 1, seed));
        all.push_back(runTrial(trainData, validation, "AdaptiveConsistency", 2, Strategy::Consistency, i + 1, seed));
        all.push_back(runTrial(trainData, validation, "AdaptivePersistence", 2, Strategy::Persistence, i + 1, seed));
    }

    for (const std::string name : {"Fixed2", "Fixed3", "Fixed4", "AdaptiveBaseline",
                                   "AdaptiveConsistency", "AdaptivePersistence"}) {
        std::vector<Result> sub;
        for (const auto& r : all)
            if (r.architecture == name) sub.push_back(r);
        report(name, sub);
    }

    writeCSV(all);
    std::cout << "\nCSV saved:\n results/E10_trials.csv\n results/E10_summary.csv\n"
              << "=== END EXPERIMENT 10 ===\n";
}
