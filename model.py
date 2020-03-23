import numpy as np
from scipy.integrate import odeint

"""
See https://github.com/alsnhll/SEIR_COVID19/blob/master/SEIR_COVID19.ipynb

Variables
 * Susceptible: Susceptible individuals (ie not yet infected nor immune)
 * Exposed: Exposed individuals - infected but not yet infectious or symptomatic
 * InfectedX individuals in severity class `X`. Severity increases with `X` and we assume individuals must pass through
 all previous classes
    * Infected1: Mild infection (hospitalization not required)
    * Infected2: Severe infection (hospitalization required)
    * Infected3: Critical infection (ICU required)
 * Recovered: individuals who have recovered from disease and are now immune
 * Dead: Dead individuals
 * Population=Susceptible+Exposed+Infected1+Infected2+Infected3+Recovered+Dead
    * Total population size (constant)

Parameters
 * InfectionRateX (beta): rate at which infected individuals in class `X` contact susceptibles and infect them
 * ExposedProgressionRate (alpha): rate of progression from the exposed to infected class
 * RecoveryRateX (gamma): rate at which infected individuals in class `X` recover from disease and become immune
 * InfectedProgressionRateX (rho): rate at which infected individuals in class `X` progress to class `X+1`
 * FatalityRate (mu): death rate for individuals in the most severe stage of disease
"""


class SeirModel:
    # case severities mapped to array indices for lookup values (eg infectionRate)
    MILD = 0
    SEVERE = 1
    CRITICAL = 2

    # current state indices
    EXPOSED = 0
    INFECTED_MILD = 1
    INFECTED_SEVERE = 2
    INFECTED_CRITICAL = 3
    RECOVERED = 4
    DEAD = 5

    def __init__(self, populationSize, incubPeriod, durMildInf, fracMild, fracSevere, fracCritical, fatalityRate,
                 timeIcuFatality, durHosp, predictionPeriod, initialState=(1, 0, 0, 0, 0, 0),
                 infectionRates=(.24, 0, 0), samplesPerDay=10):
        """
        :param populationSize (int): Number of people in population
        :param incubPeriod (int):  Incubation period, days
        :param durMildInf (int): Duration of mild infections, days
        :param fracMild (float): Fraction of infections that are mild
        :param fracSevere (float): Fraction of infections that are severe
        :param fracCritical (float): Fraction of infections that are critical
        :param fatalityRate (float): Case fatality rate (fraction of infections resulting in death)
        :param timeIcuFatality (int): Time from ICU admission to death, days
        :param durHosp (int): Duration of hospitalization, days
        :param predictionPeriod (int): Length of prediction in days
        :param initialState ([float]): Initial state, array of [exposed, infected1, infected2, infected3, recovered, dead]
        :param infectionRates ([float]): Rates of infection for [mild, severe, critical]
        :param samplesPerDay (int): Number of slices to make out of each day
        """
        self.populationSize = populationSize
        self.incubPeriod = incubPeriod
        self.durMildInf = durMildInf
        self.fracMild = fracMild
        self.fracSevere = fracSevere
        self.fracCritical = fracCritical
        self.fatalityRate = fatalityRate
        self.timeIcuFatality = timeIcuFatality
        self.durHosp = durHosp
        self.predictionPeriod = predictionPeriod
        self.initialState = initialState
        self.infectionRates = infectionRates
        self.samplesPerDay = samplesPerDay

        # calculated vars
        self.recoveryRates = np.zeros(3)
        self.infectedProgressionRates = np.zeros(3)
        self.exposedProgressionRate = 1 / self.incubPeriod

        self.fatalityRate = (1 / self.timeIcuFatality) * (self.fatalityRate / self.fracCritical)
        self.recoveryRates[2] = (1 / self.timeIcuFatality) - self.fatalityRate

        self.infectedProgressionRates[2] = (1 / self.durHosp) * (
            self.fracCritical / (self.fracCritical + self.fracSevere))
        self.recoveryRates[1] = (1 / self.durHosp) - self.infectedProgressionRates[2]

        self.recoveryRates[0] = (1 / self.durMildInf) * self.fracMild
        self.infectedProgressionRates[1] = (1 / self.durMildInf) - self.recoveryRates[0]

        self.infectionRates = np.array(self.infectionRates) / self.populationSize

    @classmethod
    def seir(cls, lastState, t, infectionRates, exposedProgressionRate, recoveryRates, infectedProgressionRates,
             fatalityRate, populationSize):
        dy = [0, 0, 0, 0, 0, 0]
        S = populationSize - sum(lastState)
        dy[cls.EXPOSED] = np.dot(infectionRates[0:2], lastState[1:3]) * S - exposedProgressionRate * lastState[0]  # E
        dy[cls.INFECTED_MILD] = exposedProgressionRate * lastState[0] - (
            recoveryRates[0] + infectedProgressionRates[1]) * lastState[1]  # I1
        dy[cls.INFECTED_SEVERE] = infectedProgressionRates[1] * lastState[1] - (
            recoveryRates[1] + infectedProgressionRates[2]) * lastState[2]  # I2
        dy[cls.INFECTED_CRITICAL] = infectedProgressionRates[2] * lastState[2] - (recoveryRates[2] + fatalityRate) * \
                                    lastState[3]  # I3
        dy[cls.RECOVERED] = np.dot(recoveryRates[0:2], lastState[1:3])  # R
        dy[cls.DEAD] = fatalityRate * lastState[cls.INFECTED_CRITICAL]  # D

        return dy

    def run(self):
        tvec = np.arange(0, self.predictionPeriod, 1 / self.samplesPerDay)
        ic = np.array(self.initialState)

        soln = odeint(self.seir, ic, tvec, args=(
            self.infectionRates, self.exposedProgressionRate, self.recoveryRates, self.infectedProgressionRates,
            self.fatalityRate, self.populationSize
        ))
        soln = np.hstack((self.populationSize - np.sum(soln, axis=1, keepdims=True), soln))
        return soln

    def getR0(self):
        # Calculate basic reproductive ratio
        return self.populationSize * (
            (
                self.infectionRates[0] / (self.infectedProgressionRates[1] + self.recoveryRates[0])
            ) + (
                self.infectedProgressionRates[1] / (self.infectedProgressionRates[1] + self.recoveryRates[0])
            ) * (
                (
                    self.infectionRates[1] / (self.infectedProgressionRates[2] + self.recoveryRates[1])
                ) + (
                    self.infectedProgressionRates[2] / (self.infectedProgressionRates[2] + self.recoveryRates[1])
                ) * (
                    self.infectionRates[2] / (self.fatalityRate + self.recoveryRates[2])
                )
            )
        )
