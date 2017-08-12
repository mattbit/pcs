const EMPTY = 0
const SANE = 1
const INFECTED = 2
const RECOVERED = 3
const IMMUNE = 4

class Simulation {
  constructor() {
    this.running = false
    this.canvas = document.getElementById('canvas')

    this.controls = {
      temperatureRange: document.getElementById('temperature'),
      temperatureOutput: document.getElementById('temperature-output'),
      stepsRange: document.getElementById('steps-per-frame'),
      runButton: document.getElementById('run-button')
    }

    this.temperature = this.controls.temperatureRange.value
    this.stepsPerFrame = this.controls.stepsRange.value
  }

  init(configuration) {
    console.log("Initializing simuation…")

    // Add listeners to controls
    this.controls.temperatureRange.addEventListener('input', this._updateTemperature.bind(this))
    this.controls.stepsRange.addEventListener('change', this._updateStepsPerFrame.bind(this))
    this.controls.runButton.addEventListener('click', this._updateRun.bind(this))

    // Create the lattice
    this.siteSize = configuration.site

    this.size = {
      width: configuration.width,
      height: configuration.height
    }

    this.lattice = new Array(this.size.width)

    // Density of population
    this.density = configuration.density

    // Fraction of population infected at start
    this.infected = configuration.infected

    // Fraction of population vaccinated
    this.immune = configuration.immune

    for (let i = 0; i < this.size.width; i++) {
      this.lattice[i] = new Array(this.size.height)

      // Initialize the sites randomly
      for (let j = 0; j < this.size.height; j++) {
        if (Math.random() < this.density) {
          let alpha = Math.random()
          if (alpha < this.immune) {
            this.lattice[i][j] = IMMUNE
          } else if (alpha < this.immune + this.infected) {
            this.lattice[i][j] = INFECTED
          } else {
            this.lattice[i][j] = SANE
          }
        } else {
          this.lattice[i][j] = EMPTY
        }
      }
    }

    canvas.width = this.size.width * this.siteSize
    canvas.height = this.size.height * this.siteSize

    let context = canvas.getContext('2d')
    this.image = context.createImageData(canvas.width, canvas.height)

    this._drawLattice()

    console.log("Lattice size: " + this.size.width + "x" + this.size.height)

    // Setup up data plot
    // this.plot = new SmoothieChart()
    // this.plot.streamTo(document.getElementById("plot"))
    //
    // this._correlation = new TimeSeries()
    // this.plot.addTimeSeries(this._correlation)
  }

  setTemperature(temperature) {
    this.temperature = temperature
  }

  _updateTemperature(e) {
    this.controls.temperatureOutput.value = e.target.value
    this.setTemperature(Number(e.target.value))
  }

  _updateRun(e) {
    if (!this.running) {
      e.target.innerText = "Pause";
      this.run()
    } else {
      e.target.innerText = "Run";
      this.pause()
    }
  }

  _updateStepsPerFrame(e) {
    this.stepsPerFrame = Number(e.target.value)
  }

  run() {
    this.running = true
    this._loop()
  }

  pause() {
    this.running = false
  }


  // This is the main loop, where Monte Carlo is performed
  _loop() {
    for (let step = 0; step < this.stepsPerFrame; step++) {
      // First we move, then propagate disease to nearest neighbours
      // using Metropolis algorithm.

      // Pick a random site
      let i = this._random(this.size.width)  // column
      let j = this._random(this.size.height) // row

      // Behaves as a random walker who moves in random direction with
      // equal probability.
      let alpha = Math.random()
      if (alpha < 0.25) {
        // Move up
        this._swap(i, j, i, j-1)
      } else if (alpha < 0.5) {
        // Move right
        this._swap(i, j, i+1, j)
      } else if (alpha < 0.75) {
        // Move down
        this._swap(i, j, i, j+1)
      } else {
        // Move left
        this._swap(i, j, i-1, j)
      }

      // Infection

      // Pick random site
      // let i = this._random(this.size.width)  // column
      // let j = this._random(this.size.height) // row

      if (this.lattice[i][j] == INFECTED) {

        let neighbours = [
          [this._enforcePBC(i+1), j],
          [this._enforcePBC(i-1), j],
          [i, this._enforcePBC(j+1)],
          [i, this._enforcePBC(j-1)]
        ]

        for(let n = 0; n < 4; n++) {
          let ii = neighbours[n][0]
          let jj = neighbours[n][1]

          if (this.lattice[ii][jj] != SANE) {
            continue
          }

          // We use Metropolis algorithm
          let alpha = Math.exp(-this._energyDelta(i, j)/this.temperature)

          if (Math.random() < alpha) {
            this._infect(ii, jj)
          }
        }

      }
    }

    // If the simulation is still running, we tell the browser to repeat
    // the _loop function as soon as the next frame is ready.
    if (this.running) {
      requestAnimationFrame(this._loop.bind(this))
    }
  }

  // Generate a random integer from 0 to max (excluded)
  _random(max) {
    return Math.floor(Math.random()*max)
  }

  // Calculates the energy delta for flipping the site
  _energyDelta(i, j) {
    // neighbours
    let left = i >= 1 ? this.lattice[i-1][j] : this.lattice[this.size.width-1][j] != INFECTED
    let right = this.lattice[(i + 1) % this.size.width][j] != INFECTED
    let top = j >= 1 ? this.lattice[i][j-1] : this.lattice[i][this.size.height-1] != INFECTED
    let bottom = this.lattice[i][(j + 1) % this.size.height] != INFECTED

    return 2.0 * this.lattice[i][j] * (top + right + bottom + left)
  }

  // Swap two sites
  _swap(i1, j1, i2, j2) {
    // Use periodic boundaries
    i2 = this._enforcePBC(i2)
    j2 = this._enforcePBC(j2)

    let v1 = this.lattice[i1][j1]
    let v2 = this.lattice[i2][j2]
    this.lattice[i2][j2] = v1
    this.lattice[i1][j1] = v2

    this._drawSite(i1, j1)
    this._drawSite(i2, j2)
  }

  // Infect a neighbour
  _infect(i, j) {
    this.lattice[i][j] = INFECTED
    this._drawSite(i, j)
  }

  // Draw a site on the canvas (i.e. updates its colour)
  _drawSite(i, j) {
    let context = this.canvas.getContext('2d')
    let color = 'white' // EMPTY by default

    if (this.lattice[i][j] == SANE) {
      color = 'black'
    } else if (this.lattice[i][j] == INFECTED) {
      color = 'red'
    } else if (this.lattice[i][j] == IMMUNE){
      color = 'blue'
    }

    context.fillStyle = color
    context.fillRect(i * this.siteSize, j * this.siteSize, this.siteSize, this.siteSize)
  }

  // Enforces periodic boundary conditions on site index
  _enforcePBC(i) {
    if (i < 0) {
      return i + this.size.width
    }

    if (i > this.size.width - 1) {
      return i % this.size.width
    }

    return i
  }

  // Draws all the sites of the lattice
  _drawLattice() {
    for (let i = 0; i < this.size.width; i++) {
      for (let j = 0; j < this.size.height; j++) {
        this._drawSite(i, j)
      }
    }
  }
}
