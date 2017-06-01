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

    for (let i = 0; i < this.size.width; i++) {
      this.lattice[i] = new Array(this.size.height)
      // Initialize the sites randomly
      for (let j = 0; j < this.size.height; j++) {
        this.lattice[i][j] = Math.random() > 0.5 ? +1 : -1
      }
    }

    canvas.width = this.size.width * this.siteSize
    canvas.height = this.size.height * this.siteSize

    let context = canvas.getContext('2d')
    this.image = context.createImageData(canvas.width, canvas.height)

    this._drawLattice()

    console.log("Lattice size: " + this.size.width + "x" + this.size.height)
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
      // Pick a random site
      let i = this._random(this.size.width)  // column
      let j = this._random(this.size.height) // row

      // We use Metropolis algorithm
      let alpha = Math.exp(-this._energyDelta(i, j)/this.temperature)

      if (Math.random() < alpha) {
        this._flip(i, j)
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
    let left = i >= 1 ? this.lattice[i-1][j] : this.lattice[this.size.width-1][j]
    let right = this.lattice[(i + 1) % this.size.width][j]
    let top = j >= 1 ? this.lattice[i][j-1] : this.lattice[i][this.size.height-1]
    let bottom = this.lattice[i][(j + 1) % this.size.height]

    return 2.0 * this.lattice[i][j] * (top + right + bottom + left)
  }

  // Flips a spin
  _flip(i, j) {
    this.lattice[i][j] *= -1
    this._drawSite(i, j)
  }

  // Draw a site on the canvas (i.e. updates its colour)
  _drawSite(i, j) {
    let context = this.canvas.getContext('2d')
    let color = this.lattice[i][j] == 1 ? 'white' : 'black'

    context.fillStyle = color
    context.fillRect(i * this.siteSize, j * this.siteSize, this.siteSize, this.siteSize)
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
