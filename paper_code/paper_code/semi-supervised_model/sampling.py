import numpy as np
from scipy import linalg
import networkx as nx
from tqdm import tqdm
import csv  # Importing csv module to save data


def generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions, random_seed=42):
    np.random.seed(random_seed)
    samplelength = sequenceLength
    initWashoutLength = sequenceLength // 10

    inputSequence = np.ones((sequenceLength, dimensions))
    outputSequence = np.ones((sequenceLength, dimensions))

    incrementsperUnit = 10
    genHistoryLength = tau * incrementsperUnit
    seed = 1.2 * np.ones((int(genHistoryLength), dimensions)) + 0.2 * (
                np.random.rand(int(genHistoryLength), dimensions) - 0.5)
    oldval = 1.2 * np.ones((1, dimensions))
    genHistory = seed
    speedup = 1

    sample = np.zeros((samplelength, dimensions))
    step = 0

    for n in range(1, samplelength + initWashoutLength + 1):
        for i in range(incrementsperUnit * speedup):
            step += 1
            tauval = genHistory[int(step % genHistoryLength)]
            newval = oldval + (0.2 * tauval / (1.0 + tauval ** 10) - 0.1 * oldval) / incrementsperUnit
            genHistory[int(step % genHistoryLength)] = oldval
            oldval = newval

        if n > initWashoutLength:
            sample[n - initWashoutLength - 1] = newval

    testseq = sample - 1.0
    outputSequence = testseq
    inputSequence[1:] = outputSequence[:-1]
    inputSequence[0] = 0.0

    return inputSequence, outputSequence


def create_small_world_matrix(size, k=4, p=0.1, weight_scale=0.5, random_seed=42):
    np.random.seed(random_seed)
    graph = nx.watts_strogatz_graph(size, k, p, seed=random_seed)
    W = nx.to_numpy_array(graph)
    W = W * (np.random.rand(size, size) - weight_scale)
    return W


def create_scale_free_matrix(size, m=3, weight_scale=0.5, random_seed=42):
    np.random.seed(random_seed)
    graph = nx.barabasi_albert_graph(size, m, seed=random_seed)
    W = nx.to_numpy_array(graph)
    W = W * (np.random.rand(size, size) - weight_scale)
    return W


def create_hybrid_matrix(size, k=4, p=0.1, m=3, alpha=0.5, small_weight_scale=0.5, scale_weight_scale=0.5,
                         random_seed=42):
    np.random.seed(random_seed)
    W_small = create_small_world_matrix(size, k, p, small_weight_scale, random_seed)
    W_scale = create_scale_free_matrix(size, m, scale_weight_scale, random_seed)
    W = alpha * W_small + (1 - alpha) * W_scale
    return W


def normalize_matrix(W, spectral_radius=1.25):
    rhoW = max(abs(linalg.eig(W)[0]))
    if rhoW > 0:
        W *= spectral_radius / rhoW
    return W


def test_reservoir(W, data, trainLen, testLen, initLen, a=0.3, reg=4.840e-07, random_seed=42):
    np.random.seed(random_seed)
    resSize = W.shape[0]
    inSize = outSize = 1

    Win = (np.random.rand(resSize, 1 + inSize) - 0.5) * 1

    X = np.zeros((1 + inSize + resSize, trainLen - initLen))
    Yt = data[None, initLen + 1:trainLen + 1]

    x = np.zeros((resSize, 1))
    for t in range(trainLen):
        u = data[t]
        x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        if t >= initLen:
            X[:, t - initLen] = np.vstack((1, u, x))[:, 0]

    A = np.dot(X, X.T) + reg * np.eye(1 + inSize + resSize)
    B = np.dot(X, Yt.T)
    Wout = np.linalg.solve(A, B).T

    Y = np.zeros((outSize, testLen))
    u = data[trainLen]
    for t in range(testLen):
        x = (1 - a) * x + a * np.tanh(np.dot(Win, np.vstack((1, u))) + np.dot(W, x))
        y = np.dot(Wout, np.vstack((1, u, x)))
        Y[:, t] = y
        u = y

    errorLen = 500
    difference = data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]
    mse = np.mean(difference ** 2)
    rmse = np.sqrt(mse)
    range_ = np.max(data[trainLen + 1:trainLen + errorLen + 1]) - np.min(data[trainLen + 1:trainLen + errorLen + 1])
    nrmse = rmse / range_

    return nrmse, mse


def optimize_reservoir_matrix(data, resSize, random_seed=42):
    np.random.seed(random_seed)
    trainLen = int(0.2 * len(data))
    testLen = len(data) - trainLen
    initLen = 100

    best_nrmse = float('inf')
    best_matrix = None
    best_params = None

    if resSize == 5:
        k_values = [2, resSize - 1]
        m_values = [2, resSize - 1]
    if resSize == 50:
        k_values = [2, 9]
        m_values = [2, 9]
    if resSize == 20:
        k_values = [2, 7]
        m_values = [2, 7]
    # if resSize == 500:
    #     k_values = [2, 9]
    #     m_values = [2, 9]
    # if resSize == 1000:
    #     k_values = [2, 11]
    #     m_values = [2, 11]

    p_values = [0.1, 0.3,0.9]

    alpha_values = [0.3, 0.9]
    small_weight_scales = [0.3,0.9]  # Weight scales for small-world network
    scale_weight_scales = [0.3, 0.9]  # Weight scales for scale-free network

    total_combinations = (len(k_values) * len(p_values) * len(m_values) *
                          len(alpha_values) * len(small_weight_scales) *
                          len(scale_weight_scales))

    print(f"Testing {total_combinations} matrix configurations...")

    results = []  # List to store parameter sets and NRMSE values

    for k in tqdm(k_values):
        for p in p_values:
            for m in m_values:
                for alpha in alpha_values:
                    for small_weight_scale in small_weight_scales:
                        for scale_weight_scale in scale_weight_scales:
                            W = create_hybrid_matrix(
                                resSize, k, p, m, alpha,
                                small_weight_scale, scale_weight_scale,
                                random_seed
                            )
                            W = normalize_matrix(W)

                            nrmses = []
                            for _ in range(3):
                                nrmse, _ = test_reservoir(W, data, trainLen, testLen, initLen, random_seed=random_seed)
                                nrmses.append(nrmse)

                            avg_nrmse = np.mean(nrmses)

                            if avg_nrmse < best_nrmse:
                                best_nrmse = avg_nrmse
                                best_matrix = W.copy()
                                best_params = {
                                    'k': k,
                                    'p': p,
                                    'm': m,
                                    'alpha': alpha,
                                    'small_weight_scale': small_weight_scale,
                                    'scale_weight_scale': scale_weight_scale,
                                    'nrmse': avg_nrmse
                                }
                                print(f"\nNew best configuration found!")
                                print(f"NRMSE: {best_nrmse:.6f}")
                                print(f"Parameters: {best_params}")

                            # Append the current configuration and NRMSE to the results list
                            results.append({
                                'k': k,
                                'p': p,
                                'm': m,
                                'alpha': alpha,
                                'small_weight_scale': small_weight_scale,
                                'scale_weight_scale': scale_weight_scale,
                                'nrmse': avg_nrmse
                            })

    # Save the results to a CSV file
    with open(f'parameter_nrmse_results_n=={resSize}_RS={random_seed}_new_errorLen=500_all_random_seed.csv', mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    return best_matrix, best_params


if __name__ == "__main__":
    sequenceLength = 10000
    tau = 29
    dimensions = 1
    for iiii in tqdm(range(42,52,1)):
        for iiiii in [5,50,20]:
            o2 = iiiii
            o1 = iiii
            random_seed = o1

            _, outputSequence = generate_multi_dimensional_MG_sequence(
                sequenceLength, tau, dimensions, random_seed=random_seed
            )
            data = outputSequence.squeeze()

            best_W, best_params = optimize_reservoir_matrix(data, resSize=o2,random_seed=random_seed)
            print("\nOptimization complete!")
            print("Best parameters:", best_params)
