import random

# ? Should we be making some mutations that work on Permutation Encoding 
# ? specifically.

# Inspiration for the functions could be found in the paper:
#
# Nitasha Soni et al, / (IJCSIT) International Journal of Computer Science
# and Information Technologies, Vol. 5 (3) , 2014, 4519-4521 which can be
# found at http://ijcsit.com/docs/Volume%205/vol5issue03/ijcsit20140503404.pdf
# Could also cite https://arxiv.org/pdf/1203.3099.pdf

# Defines the dictionary of possible mutations and the corresponding function
# pointers, and this can easily be used in the core library to use the string 
# to run the function.


def insert_mutation(genome):
    """Performs an insert mutation operation on the provided genome. First,
    picks two alleles at random and moves the second allele to be right after
    the first allele and shifting the rest of the genome to accommodate the
    change. This can be performed on any type of genome, but all the values in
    the genome must be of the same type.

    Args:
        genome (required): The genome that need to be mutated.

    Returns:
        The mutated genome
    """

    alleleA = random.randint(0, len(genome) - 1)
    alleleB = random.randint(0, len(genome) - 1)

    # Alleles should not be the same
    while alleleA != alleleB:
        alleleB = random.randint(0, len(genome) - 1)

    # alleleA needs be always less than alleleB. Swap if needed
    if alleleA > alleleB:
        temp_allele = alleleA
        alleleA = alleleB
        alleleB = temp_allele

    alleleA_val = genome[alleleA]

    for shift in range(alleleA + 1, alleleB + 1):
        genome[shift - 1] = genome[shift]

    genome[alleleB] = alleleA_val

    return genome


def flip_mutation(genome, total_flips):
    """Flip mutation can only be performed on a genome that is binary in
    nature. This operator randomly selects total_flips number of points and
    flips the bits at those positions in the genome. If total_flips is 1 then
    the mutation is called single-point mutation and if it is greater than 1
    it is called multiple-point mutation.

    Args:
        genome (A boolean array / list): The genome that is to be mutated.
        total_flips (int): The number of flips that need to be performed.

    Returns:
        Boolean array / list: The mutated genome.
    """

    # Raise an exception of the data type of the genome is not boolean
    if not all(isinstance(allele, bool) for allele in genome):
        # TODO: Error handling
        pass

    for _ in range(total_flips):
        point = random.randint(0, len(genome))

        genome[point] = not genome[point]

    return genome


# ! should this function be restricted to binary values?
def interchanging_mutation(genome):
    """Performs Interchanging Mutation on the given genome. Two position
    are randomly chosen and they values corresponding to these positions
    are interchanged.

    Args:
        genome (required): The genome that is to be mutated.

    Returns:
        The mutated genome.
    """
    alleleA = random.randint(0, len(genome))
    alleleB = random.randint(0, len(genome))

    # Alleles should not be the same
    while alleleA != alleleB:
        alleleB = random.randint(0, len(genome))

    temp_gene = genome[alleleA]
    genome[alleleA] = genome[alleleB]
    genome[alleleB] = temp_gene

    return genome


def reversing_mutation(genome):
    """Performs a reversing mutation to the given genome. A random pivot
    point is chosen and the values post that point are reversed.

    Args:
        genome (required): The genome that is to be mutated.

    Returns:
        The mutated genome.
    """
    reverse_point = random.randint(0, len(genome))
    mid_point = reverse_point + (len(genome) - reverse_point) / 2

    for allele in range(reverse_point, mid_point):
        temp_allele = genome[allele]
        genome[allele] = genome[2 * mid_point - allele]
        genome[2 * mid_point + allele] = temp_allele

    return genome


def uniform_mutation(genome, lower_bound, upper_bound):
    """Performs an uniforrm mutation on the given genome. This operator
    chooses a random allele in the genome and changes its value to a 
    uniformly-randomly generated value between upper and lower bound as
    provided. This function only works on integer or float representations.

    Args:
        genome (required): The genome to be mutated.
        upper_bound (required): The upper bound to the Uniform(a, b) random generator.
        lower_bound (required): The lower bound to the Uniform(a, b) random generator.

    Returns:
        Returns the mutated genome.
    """
    if not all(isinstance(gene, [float, int]) for gene in genome):
        # TODO: Handle the type coercion exception.
        pass

    allele = random.randint(0, len(genome))

    genome[allele] = random.uniform(lower_bound, upper_bound)

    return genome


def creep_mutation(
    genome, distribution, 
    alpha=None, beta=None,
    lambd=None, mu=None,
    sigma=None, kappa=None
):
    """This mutation operator performs a creep mutation on the genome.
    Taking an additional argument than the uniform_mutation this is a 
    generalized version of the same. Instead of using an Uniform(a, b)
    distribution to sample the random number from, it uses the distribution
    that is provided by the user. The following distributions are available,
    the distribution should be as the value inside quotes.

    "beta": Beta distribution. alpha and beta are required. And alpha > 0
            and beta > 0.

    "exp": Exponential distribution. lambd is required. It is called lambd as
           lambda is a reserved keyword in python.

    "gamma": Gamma distribution. alpha and beta are required. And alpha > 0 
             and beta > 0.

    "gauss": Gaussian distribution. mu and sigma are required.

    "lognorm": Log normal distribution. mu and sigma are required.

    "vonmises": The Von Mises distribution. mu is the mean angle and kappa is 
                the concentration parameter. Here, kappa >= 0 and 0 < pi < 2 * pi.

    "pareto": Pareto distribution. alpha is required and is the shape parameter.

    "weibull": Weibull distribution. alpha and beta are required. Where alpha
               is the scale parameter and beta is the shape parameter.

    Args:
        genome (required): The genome that is to be mutated.
        alpha (optional): One of the parameters. All parameters are explained above.
        beta (optional): One of the parameters. All parameters are explained above.
        lambda (optional): One of the parameters. All parameters are explained above.
        distribution (string): The distribution function to be used in the mutation.
    """

    allele_pos = random.randint(0, len(genome))

    if distribution == "beta":
        # Beta Distribution.
        # Check for parameters provided
        if alpha is None:
            # TODO: Raise corresponding exception
            pass

        if beta is None:
            # TODO: Raise corresponding exception
            pass

        genome[allele_pos] = random.betavariate(alpha, beta)
    elif distribution == "exp":
        # Exponential Distribution
        # Check for parameters provided
        if lambd is None:
            # TODO: Raise corresponding exception
            pass

        genome[allele_pos] = random.expovariate(lambd)
    elif distribution == "gamma":
        # Gamma Distribution
        # Check for parameters provided
        if alpha is None:
            # TODO: Raise corresponding exception
            pass

        if beta is None:
            # TODO: Raise corresponding exception
            pass

        genome[allele_pos] = random.gammavariate(alpha, beta)
    elif distribution == "gauss":
        # Gaussian Distribution
        # Check for parameters provided
        if mu is None:
            # TODO: Raise corresponding exception
            pass

        if sigma is None:
            # TODO: Raise corresponding exception
            pass

        genome[allele_pos] = random.gauss(mu, sigma)
    elif distribution == "lognorm":
        # Log Norrmal Distribution
        # Check for parameters provided
        if mu is None:
            # TODO: Raise corresponding exception
            pass

        if sigma is None:
            # TODO: Raise corresponding exception
            pass

        genome[allele_pos] = random.lognormvariate(mu, sigma)
    elif distribution == "vonmises":
        # Von Mises Distribution
        # Check for parameters provided
        if mu is None:
            # TODO: Raise corresponding exception
            pass
        
        if kappa is None:
            # TODO: Raise corresponding exception
            pass
        
        genome[allele_pos] = random.vonmisesvariate(mu, kappa)
    elif distribution == "pareto":
        # Pareto Distribution
        # Check for parameters provided
        if alpha is None:
            # TODO: Raise corresponding exception
            pass
        
        genome[allele_pos] = random.paretovariate(alpha)
    elif distribution == "weibull":
        # Weibull Distribution
        # Check for parameters provided
        if alpha is None:
            # TODO: Raise corresponding exception
            pass

        if beta is None:
            # TODO: Raise corresponding exception
            pass

        genome[allele_pos] = random.weibullvariate(alpha, beta)
    else:
        # TODO: Handle the exception of the provided random distribution not
        # TODO: being available.
        pass

    return genome


mutation_strats = {
    "insert"        :   insert_mutation,
    "flip"          :   flip_mutation,
    "interchanging" :   interchanging_mutation,
    "reversing"     :   reversing_mutation,
    "uniform"       :   uniform_mutation,
    "creep"         :   creep_mutation
}
