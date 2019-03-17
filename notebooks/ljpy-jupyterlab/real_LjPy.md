1. Make sure Jupyter Lab theme is dark, close all open tabs
* Open Jupyter Lab:
    1. Go to `/home/dror/soft/jlpython3`
    * Run `pipenv shell`
    * Go to `../notebooks/ljpy`
    * Run `jupyter lab`
    * Test both themes for visibility at site
* Set timer for talk:
    1. Click File->New->Console
    * Write `for _ in tqdm.`
    * Press Tab
    * Choose `trange`
    * Add `(17`
    * Say you've written this before
    * Press up to get history
    * Evaluate
    * Put tab below
* Introduce self:
    1. Worked with python for about fifteen years
    * Scripts for mathemtical research, mostly with sage
    * Scripts for automated music analysis and djing
    * Currently at Bugatone, working with sensors
    * Say forgot to put Bugatone logo
    * Look for Bugatone logo file and open it
    * Add image to markdown
    * Married to a Slovenian, moved to Ljubljana a year and a half ago
* Starting history before Jupyter Lab:
    1. I'll talk a bit about stuff before and leading to Jupyter Lab
    * Completion, showed in console tab
    * Input history, showed in console tab
    * Customizable GUI, showed in themes
    * Multiple views, for example two views on same markdown file
    * Inlined plots:
        1. Open `inline_plots.ipynb`
        * Evaluate first two cells
        * OMG Inlined plots :)
    * Visual supremacy, mix of text, code, and output in `inline_plots`
    * Collaboration supremacy, can send the notebook file to others
    * Widgets:
        1. Open `widgets.ipynb`
        * Evaluate cells to show plotly, use slider
    * Rich output, evaluate pandas cells
    * Showing multiple kernels:
        1. Json file business:
            1. Open json file with double click
            * Say SAGE_ROOT is missing
            * Open json file with editor
            * Edit json file
            * Point to updated json file in json viewer
        * Open SageMath:
            1. Click File->New->Console
            * Choose SageMath kernel
            * Define a number field
        * Open console of Xeus-Cling-C++14
            1. Enter `#include <iostream>`
            * Enter `std::cout << "Hello, World!\n";`
* Jupyter Lab:
    1. Many file types, already shown images and json
    * Themes, chose theme at beginning
    * Multiple tab view, a few tabs are actually open
    * Terminal, already open
    * Console, showed console a few times
    * Inspector:
        1. Open a new python console
        * Open Inspector from command pallete
        * Start writing `np.`
        * If nothing happends, import numpy
        * Show beatiful inspector. Continue to `np.linalg.norm`
    * Multiple views on same kernel:
        1. Open a new python console
        * Enter `walk = 2 * np.random.randint(0, 2, 100) - 1`
        * `plt.plot(walk)`
        * `%matplotlib inline`
        * `plt.plot(walk)`
        * Forgot to do cumulative sum
        * `plt.plot(np.cumsum(walk))`
        * Open new Notebook with same kernel
        * Define function for walks:
```python
def random_walks(size):
              steps = 2 * np.random.randint(0, 2, size) - 1
              return np.cumsum(steps)
```
        * `walks = random_walks((2, 100))`
        * `plt.plot(walks)`
        * Fix function to use `axis=-1`
        * `walks = random_walks((2, 100))`
        * `walks.shape`
        * `plt.plot(walks.T)`
    * Extensions:
        1. Matplotlib, didn't show, but it's not as cool as plotly
        * Plotly, already shown sine waves
        * Beakerx, shown for pandas; lots of other features
        * Drawio:
            1. Open new Drawio diagram
            * Create new rectangle
            * Enter text "Initialize super important program"
            * Create new rectangle
            * Enter text "print Hello World"
            * Create arrow between the two
            * Enter arrow text "some super important event?"
        * Status bar, point to status bar
        * Variable inspector:
            1. Open console
            * Enter `x = 1`
            * `y = [2, 3, 4]`
            * Right click and select variable inspector
            * Say I have a version that shows correct stack information during pdb, in the works...
        * Github:
            1. If there's internet
            * Open GitHub sidebar
            * Open my user
            * Go to autoimport and recall how no imports were used
            * Open console
            * Run `os` and point to AutoImport line
            * Run `os.wolk("/home/dror")` and point to suggestion
            * Go to suggestions github folder
            * Enter in console `%findsymbol pearson` and press tab
