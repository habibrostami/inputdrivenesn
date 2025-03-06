import time
import numpy as np
import torch
start_time = time.time()
random_seed = 10
# Set random seeds for reproducibility
np.random.seed(random_seed)  # Set seed for NumPy
torch.manual_seed(random_seed)  # Set seed for PyTorch
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(random_seed)  # Set seed for all GPUs (if applicable)

# Generate MG time series
def generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions):
    samplelength = sequenceLength
    initWashoutLength = sequenceLength // 10

    inputSequence = np.ones((sequenceLength, dimensions))
    outputSequence = np.ones((sequenceLength, dimensions))

    incrementsperUnit = 10
    genHistoryLength = tau * incrementsperUnit
    seed = 1.2 * np.ones((int(genHistoryLength), dimensions)) + 0.2 * (np.random.rand(int(genHistoryLength), dimensions) - 0.5)
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

# Example usage:
sequenceLength = 20000
tau = 29
dimensions = 1
inputSequence, outputSequence = generate_multi_dimensional_MG_sequence(sequenceLength, tau, dimensions)
print("Input Sequence Shape:", inputSequence.shape)
print("Output Sequence Shape:", outputSequence.shape)
import matplotlib.pyplot as plt

# Plotting the Mackey-Glass time series generated above
plt.figure(figsize=(10, 6))
plt.plot(outputSequence[:500], label="Mackey-Glass Time Series")
plt.title("Mackey-Glass Time Series")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()



#------------------------------------------------------------------------------------------


import numpy as np
def generate_NARMA_sequence(sequenceLength, memoryLength):
    # Create input
    inputSequence = np.column_stack((np.ones(sequenceLength), np.random.rand(sequenceLength)))
    # Use the input sequence to drive a NARMA equation
    outputSequence = 0.1 * np.ones(sequenceLength)
    for i in range(memoryLength, sequenceLength):
        # Insert suitable NARMA equation, this is just an ad hoc example
        outputSequence[i] = 0.7 * inputSequence[i - memoryLength, 1] + 0.1 \
            + (1 - outputSequence[i - 1]) * outputSequence[i - 1]
    return inputSequence, outputSequence
# Example usage
sequenceLength = 1000
memoryLength = 10
inputSequence, outputSequence = generate_NARMA_sequence(sequenceLength, memoryLength)

# Plot the NARMA time series
plt.figure(figsize=(10, 6))
plt.plot(outputSequence[:500], label="NARMA Time Series")
plt.title("NARMA Time Series")
plt.xlabel("Time Step")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


#------------------------------------------------------------------------------------------------

