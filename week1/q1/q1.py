import json
import numpy as np
import matplotlib.pyplot as plt

def inv_transform(distribution: str, num_samples: int, **kwargs) -> list:
    """ populate the 'samples' list from the desired distribution """

    samples = []

    # TODO: first generate random numbers from the uniform distribution
    uniform_samples = np.random.uniform(0, 1, num_samples)

    rounded_samples = np.round(uniform_samples, 4)
   

    if distribution == "cauchy":
        samples = np.tan(np.pi * (rounded_samples - 0.5)).tolist()
    elif distribution == "exponential":
        samples = (-np.log(1 - rounded_samples) / kwargs.get("lambda")).tolist()
    # END TODO
            
    return samples


if __name__ == "__main__":
    np.random.seed(42)

    for distribution in ["cauchy", "exponential"]:
        file_name = "q1/q1_" + distribution + ".json"
        args = json.load(open(file_name, "r"))
        samples = inv_transform(**args)
        
        with open("q1_output_" + distribution + ".json", "w") as file:
            json.dump(samples, file)

        # TODO: plot and save the histogram to "q1_" + distribution + ".png"
        plt.hist(samples, bins=100, alpha=0.7, density = False)
        plt.title(f"Plot for {distribution} distribution")
        plt.xlabel("Value")
        plt.ylabel("Frequency")
        plt.savefig("q1_" + distribution + ".png")
        plt.clf()
        # END TODO
