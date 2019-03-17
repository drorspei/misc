1. Make sure Jupyter Lab theme is dark, close all open tabs
2. Open Jupyter Lab:
    1. Go to `/home/dror/soft/jlpython3`
    2. Run `pipenv shell`
    3. Go to `../notebooks/ljpy`
    4. Run `jupyter lab`
    5. Test both themes for visibility at site
3. Set timer for talk:
    1. Click File->New->Console
    2. Write `for _ in tqdm.`
    3. Press Tab
    4. Choose `trange`
    5. Add `(17`
    6. Say you've written this before
    7. Press up to get history
    8. Evaluate
    9. Put tab below
4. Introduce self:
    1. Worked with python for about fifteen years
    2. Scripts for mathemtical research, mostly with sage
    3. Scripts for automated music analysis and djing
    4. Currently at Bugatone, working with sensors
    5. Say forgot to put Bugatone logo
    6. Look for Bugatone logo file and open it
    7. Add image to markdown
    8. Married to a Slovenian, moved to Ljubljana a year and a half ago
5. Starting history before Jupyter Lab:
    1. I'll talk a bit about stuff before and leading to Jupyter Lab
    2. Completion, showed in console tab
    3. Input history, showed in console tab
    4. Customizable GUI, showed in themes
    5. Multiple views, for example two views on same markdown file
    6. Inlined plots:
        1. Open `inline_plots.ipynb`
        2. Evaluate first two cells
        3. OMG Inlined plots :)
    7. Visual supremacy, mix of text, code, and output in `inline_plots`
    8. Collaboration supremacy, can send the notebook file to others
    9. Widgets:
        1. Open `widgets.ipynb`
        2. Evaluate cells to show plotly, use slider
    10. Rich output, evaluate pandas cells
    11. Showing multiple kernels:
        1. Json file business:
            1. Open json file with double click
            2. Say SAGE_ROOT is missing
            3. Open json file with editor
            4. Edit json file
            5. Point to updated json file in json viewer
        2. Open SageMath:
            1. Click File->New->Console
            2. Choose SageMath kernel
            3. Define a number field
        3. Open console of Xeus-Cling-C++14
            1. Enter `#include <iostream>`
            2. Enter `std::cout << "Hello, World!\n";`
6. Jupyter Lab:
    1. Many file types, already shown images and json
    2. Themes, chose theme at beginning
    3. Multiple tab view, a few tabs are actually open
    4. Terminal, already open
    5. Console, showed console a few times
    6. Inspector:
        1. Open a new python console
        2. Open Inspector from command pallete
        3. Start writing `np.`
        4. If nothing happends, import numpy
        5. Show beatiful inspector. Continue to `np.linalg.norm`
    7. Multiple views on same kernel:
        1. Open a new python console
        2. Enter `walk = 2 * np.random.randint(0, 2, 100) - 1`
        3. `plt.plot(walk)`
        4. `%matplotlib inline`
        5. `plt.plot(walk)`
        6. Forgot to do cumulative sum
        7. `plt.plot(np.cumsum(walk))`
        8. Open new Notebook with same kernel
        9. Define function for walks:
            ```python
            def random_walks(size):
                          steps = 2 * np.random.randint(0, 2, size) - 1
                          return np.cumsum(steps)
            ```
        10. `walks = random_walks((2, 100))`
        11. `plt.plot(walks)`
        12. Fix function to use `axis=-1`
        13. `walks = random_walks((2, 100))`
        14. `walks.shape`
        15. `plt.plot(walks.T)`
    8. Extensions:
        1. Matplotlib, didn't show, but it's not as cool as plotly
        2. Plotly, already shown sine waves
        3. Beakerx, shown for pandas; lots of other features
        4. Drawio:
            1. Open new Drawio diagram
            2. Create new rectangle
            3. Enter text "Initialize super important program"
            4. Create new rectangle
            5. Enter text "print Hello World"
            6. Create arrow between the two
            7. Enter arrow text "some super important event?"
        5. Status bar, point to status bar
        6. Variable inspector:
            1. Open console
            2. Enter `x = 1`
            3. `y = [2, 3, 4]`
            4. Right click and select variable inspector
            5. Say I have a version that shows correct stack information during pdb, in the works...
        7. Github:
            1. If there's internet
            2. Open GitHub sidebar
            3. Open my user
            4. Go to autoimport and recall how no imports were used
            5. Open console
            6. Run `os` and point to AutoImport line
            7. Run `os.wolk("/home/dror")` and point to suggestion
            8. Go to suggestions github folder
            9. Enter in console `%findsymbol pearson` and press tab
