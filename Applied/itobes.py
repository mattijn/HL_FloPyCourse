import numpy as np
import pandas
import geopandas as gpd
import matplotlib.pyplot as plt
from shapely.geometry import box
import flopy
from flopy.utils.reference import SpatialReference


class BronData(object):
    """
    Represent an object that must be filled with information to build a Modflow model
    """

    def __init__(
        self,
    ):
        """
        Construct an empty BronData class
        """
        self.source = {}

    def cellsize_large(self, value=100):
        """
        Define grid parameters

        Parameters
        ----------
        value : int
            cellsize in meters for large grid-cells
        """
        self.source.update({"cellsize_large": value})

    def cellsize_small(self, value=10):
        """
        Define grid parameters

        Parameters
        ----------
        value : int
            cellsize in meters for large grid-cells
        """
        self.source.update({"cellsize_small": value})

    def offset_large(self, value=1000):
        """
        Define grid parameters

        Parameters
        ----------
        value : int
            offset in meters for large grid-cells
        """
        self.source.update({"offset_large": value})

    def offset_small(self, value=50):
        """
        Define grid parameters

        Parameters
        ----------
        value : int
            offset in meters for small grid-cells
        """
        self.source.update({"offset_small": value})

    def years_total_run(self, value=100):
        """
        Define stress periods and time-steps

        Parameters
        ----------
        value = int
            total run in years
        """
        self.source.update({"years_total_run": value})

    def months_summer(self, value=10):
        """
        Define stress periods and time-steps

        Parameters
        ----------
        value : int
            cellsize in meters for large grid-cells
        """
        self.source.update({"months_summer": value})

    def months_winter(self, value=1000):
        """
        Define stress periods and time-steps

        Define grid parameters

        Parameters
        ----------
        value : int
            offset in meters for large grid-cells
        """
        self.source.update({"months_winter": value})

    def days_in_month(self, value=50):
        """
        Define stress periods and time-steps

        Parameters
        ----------
        value : int
            offset in meters for small grid-cells
        """
        self.source.update({"days_in_month": value})

    def days_initization(self, value=50):
        """
        Define stress periods and time-steps

        Parameters
        ----------
        value : int
            offset in meters for small grid-cells
        """
        self.source.update({"days_initization": value})

    def start_in_summer(self, value=True):
        """
        Define stress periods and time-steps

        Parameters
        ----------
        value : boolean
            True or False if first stress-period, after initiziation is summer
        """
        self.source.update({"start_in_summer": value})

    def name_modelrun(self, value):
        """
        Define modelname and modellocation

        Parameters
        ----------
        value : string
            name of the modelrun
        """
        self.source.update({"modelname": value})

    def location_exe(self, value="../Exe/mf2005.exe"):
        """
        Define modelname and modellocation

        Parameters
        ----------
        value : string
            location of the executable of modflow
        """
        self.source.update({"exe_name": value})

    def location_workspace(self, value="Results"):
        """
        Define modelname and modellocation

        Parameters
        ----------
        value : string
            location of the executable of modflow
        """
        self.source.update({"model_ws": value})


class BronModel(object):
    """
    Represent an object that contains the functions required to build a Modflow model
    """

    def __init__(self, soil_df, well_gdf, bron_data):
        """
        Class to construct a BronModel
        """
        self.dis_params(soil_df, well_gdf, bron_data.source)
        self.init_model(bron_data.source)

        self.lpf_params(soil_df)
        self.bas_params()
        self.wel_params(well_gdf, bron_data.source)
        
        self.build_model(bron_data.source)

    def offset_wells(self, well_gdf, offset):
        """
        This functions takes as input the located wells and places a buffer around 
        it based on the offset given as input
        
        Parameters
        ----------
        well_gdf : geodataframe
            geodataframe containing the well defintions
        offset : int
            offset in meters
        
        Returns
        -------
        tuple
            bounding box defining minx, miny, maxx, maxy including offset
        """

        minx, miny, maxx, maxy = well_gdf.total_bounds
        offset_bbox = box(minx, miny, maxx, maxy).buffer(offset, join_style=2).bounds
        return offset_bbox

    def array_generator_column(
        self, bbox_large, bbox_small, cellsize_large, cellsize_small
    ):
        """
        This functions computes the irregular column array to be used for a 
        modflow grid definition
        
        Parameters
        ----------
        bbox_large : tuple
            bounding box defintion of the large cells
        bbox_small : tuple
            bounding box defintion of the centered small cells
        cellsize_large : int        
            cellsize in meters of the large cells
        cellsize_small : int
            cellsize in meters of the small cells                        
        
        Returns
        -------
        numpy.array
            vector array of the column defintion of a modflow grid
        """        

        delr = np.array([])

        while delr.sum() + bbox_large[0] < bbox_small[0] - cellsize_large:
            delr = np.append(delr, [cellsize_large])
        while (
            delr.sum() + bbox_large[0] > bbox_small[0] - cellsize_large
            and delr.sum() + bbox_large[0] < bbox_small[2] + cellsize_large
        ):
            delr = np.append(delr, [cellsize_small])
        while (
            delr.sum() + bbox_large[0] > bbox_small[2] + cellsize_large
            and delr.sum() + bbox_large[0] < bbox_large[2]
        ):
            delr = np.append(delr, [cellsize_large])
        return delr

    def array_generator_row(
        self, bbox_large, bbox_small, cellsize_large, cellsize_small
    ):
        """
        This functions computes the irregular row array to be used for a 
        modflow grid definition
        
        Parameters
        ----------
        bbox_large : tuple
            bounding box defintion of the large cells
        bbox_small : tuple
            bounding box defintion of the centered small cells
        cellsize_large : int        
            cellsize in meters of the large cells
        cellsize_small : int
            cellsize in meters of the small cells                        
        
        Returns
        -------
        numpy.array
            vector array of the row defintion of a modflow grid
        """         

        delc = np.array([])

        while bbox_large[3] - delc.sum() > bbox_small[3] + cellsize_large:
            delc = np.append(delc, [cellsize_large])
        while (
            bbox_large[3] - delc.sum() < bbox_small[3] + cellsize_large
            and bbox_large[3] - delc.sum() > bbox_small[1] - cellsize_large
        ):
            delc = np.append(delc, [cellsize_small])
        while (
            bbox_large[3] - delc.sum() < bbox_small[1] - cellsize_large
            and bbox_large[3] - delc.sum() > bbox_large[1]
        ):
            delc = np.append(delc, [cellsize_large])
        return delc

    def dis_params(self, soil_df,well_gdf, bron_data):
        """
        This functions prepares the used parameters for the modflow DIS package
        
        Parameters
        ----------
        soil_df : dataframe
            pandas dataframe containing the soil-schematization
        well_gdf : geodataframe
            geopandas geodataframe containing the well-defintitions
        bron_data : dict
            dictionary containing data to build up the model
        """          

        bbox_small = self.offset_wells(well_gdf, bron_data["offset_small"])
        bbox_large = self.offset_wells(well_gdf, bron_data["offset_large"])

        delr = self.array_generator_column(
            bbox_large,
            bbox_small,
            bron_data["cellsize_large"],
            bron_data["cellsize_small"],
        )
        delc = self.array_generator_row(
            bbox_large,
            bbox_small,
            bron_data["cellsize_large"],
            bron_data["cellsize_small"],
        )

        nrow = delc.shape[0]
        ncol = delr.shape[0]

        nlays = soil_df.shape[0]
        top = soil_df.iloc[0]["from [m-gl]"]
        botm = (soil_df["until [m-gl]"] * -1).tolist()

        # define no. of stress periods in model
        # two seasons * no. years + initization period = no. model stress periods
        nper = 2 * bron_data["years_total_run"] + 1

        # define period lengths for each stress period
        stress_period_year = [
            bron_data["months_summer"] * bron_data["days_in_month"],
            bron_data["months_winter"] * bron_data["days_in_month"],
        ]
        if not bron_data["start_in_summer"]:
            stress_period_year = stress_period_year[::-1]
        perlen = [bron_data["days_initization"]] + stress_period_year * bron_data[
            "years_total_run"
        ]

        # define time steps for each stress period
        # initization period is a single time step. (steady-state stress period)
        nstp = [1] + [
            int(stress_period / bron_data["days_in_month"])
            for idx, stress_period in enumerate(perlen)
            if idx > 0
        ]

        # define steady state for for each stress period
        # if a stress period is in transient phase it can have fluctuating objects, 
        # such as pumping rate wells/inflow etc.
        # in our case only the first period is in steady state (True) and others are transient (False)
        steady = np.zeros(nper, dtype=bool)
        steady[0] = True

        # assign results to self
        self.dis = {}
        self.dis.update(
            {
                "bbox_large": bbox_large,
                "bbox_small": bbox_small,
                "nrow": nrow,
                "ncol": ncol,
                "delr": delr,
                "delc": delc,
                "nlays": nlays,
                "top": top,
                "botm": botm,
                "nper": nper,
                "perlen": perlen,
                "nstp": nstp,
                "steady": steady,
            }
        )

    def lpf_params(self, df):
        """
        This functions prepares the used parameters for the modflow LPF package
        
        Parameters
        ----------
        df : dataframe
            pandas dataframe containing the soil-schematization
        """          

        # define hydraulic conductivity for all layers based on column 'permeability [m/d]'
        Kx = np.zeros((self.mf.dis.nlay, self.mf.dis.nrow, self.mf.dis.ncol))
        for ix in range(self.mf.dis.nlay):
            Kx[ix, :, :] = df.iloc[ix]["permeability [m/d]"]

        # define confined layers, where c-value is smaller than 1, else are convertible
        laytyp = (~(df["c-value [d]"] > 1).values).astype(int)

        self.lpf = {}
        self.lpf.update({"hk": Kx, "laytyp": laytyp})

    def bas_params(self):
        """
        This functions prepares the used parameters for the modflow BAS package
        """          
        # configuration of active zones and initial heads: BAS package
        ibound = np.ones((self.mf.dis.nlay, self.mf.dis.nrow, self.mf.dis.ncol), dtype=np.int32)
#         ibound[:, :, 0] = -1
#         ibound[:, :, -1] = -1        

        # initial start head levels
        strt = np.zeros((self.mf.dis.nlay, self.mf.dis.nrow, self.mf.dis.ncol), dtype=np.float32)
        strt[:, :, 0] = 1.5
#        strt[:, :, -1] = 0.        
        
        self.bas = {}
        self.bas.update({"ibound": ibound, "strt": strt})

    def wel_params(self, well_gdf, bron_data):
        """
        This functions prepares the used parameters for the modflow WEL package
        
        Parameters
        ----------
        well_gdf : geodataframe
            geopandas geodataframe containing the well-defintitions
        bron_data : dict
            dictionary containing data to build up the model
        """            

        # we create a PlotMapView object so we can use the .mg.intersect function to get the i, j
        # grid index value of the wells. 
        # since the PlotMapView object returns a matplotlib figure, we add our wells to the 
        # grid defintion for visualization purposes
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(1, 1, 1, aspect="equal")
        modelmap = flopy.plot.PlotMapView(model=self.mf, ax=ax)
        linecollection = modelmap.plot_grid(linewidth=0.5, color="royalblue")
        well_gdf.plot(ax=ax)

        # collect row, column indices from well coordinates
        i_vals_r, j_vals_c = list(
            zip(
                *[
                    modelmap.mg.intersect(row.x, row.y)
                    for idx, row in well_gdf.iterrows()
                ]
            )
        )

        # assign into well_gdf
        well_gdf["ix_row"] = i_vals_r
        well_gdf["ix_col"] = j_vals_c

        if bron_data["start_in_summer"]:
            stress_periods_summer = list(range(1, len(self.mf.dis.perlen.array), 2))
        else:
            stress_periods_summer = list(range(2, len(self.mf.dis.perlen.array), 2))
        
        # define well behaviour for each stress-period
        well_spd = {}
        for idx, val in enumerate(self.mf.dis.perlen.array):
            if idx == 0:
                # initization stress period
                # wells are not pumping on steady state (first stress period),
                # but we have to insert zero pumping rate to at least one cells
                well_spd[idx] = [
                    0,
                    0,
                    0,
                    0,
                ]  
            elif idx in stress_periods_summer:
                # stress periods for summer
                # [v,  r,  c, volume]
                # v = index vertical layer
                # r = index row
                # c = index column
                # v = volume m3/day (m3_hr * hr_day active, defaults to full day)
                well_spd[idx] = [
                    [
                        row.index_layer,
                        row.ix_row,
                        row.ix_col,
                        row.m3_hr_summer * row.hr_day,
                    ]
                    for idx, row in well_gdf.iterrows()
                ]
            else:
                # stress periods for winter
                well_spd[idx] = [
                    [
                        row.index_layer,
                        row.ix_row,
                        row.ix_col,
                        row.m3_hr_winter * row.hr_day,
                    ]
                    for idx, row in well_gdf.iterrows()
                ]
        self.wel = {}
        self.wel.update({"stress_period_data": well_spd})

    def init_model(self, bron_data):
        """
        Initiziation of the model starts with the Modflow object and spatial and 
        temporal discretization parameters of the DIS package
        
        Parameters
        ----------
        bron_data : dict
            dictionary containing data to build up the model        
        """
        
        mf = flopy.modflow.Modflow(
            bron_data["modelname"],
            exe_name=bron_data["exe_name"],
            model_ws=bron_data["model_ws"],
        )  # --> needed when using UPW insted of LPF, version='mfnwt')

        # apply the spatial and temporal discretization parameters to the DIS package
        mf_dis = flopy.modflow.ModflowDis(
            mf,
            nlay=self.dis["nlays"],
            nrow=self.dis["nrow"],
            ncol=self.dis["ncol"],
            delr=self.dis["delr"],
            delc=self.dis["delc"],
            top=self.dis["top"],
            botm=self.dis["botm"],
            nper=self.dis["nper"],
            perlen=self.dis["perlen"],
            nstp=self.dis["nstp"],
            steady=self.dis["steady"],
        )        

        mf.dis.sr = SpatialReference(
            delr=self.dis["delr"],
            delc=self.dis["delc"],
            xul=self.dis["bbox_large"][0],
            yul=self.dis["bbox_large"][3],
            epsg=28992,
        )

        mf.modelgrid.set_coord_info(
            xoff=self.dis["bbox_large"][0],
            yoff=self.dis["bbox_large"][3] - self.dis["delc"].sum(),
            epsg=28992,
        )

        self.mf = mf

    def build_model(self, bron_data):
        """
        Extending the intiziatised model with the other packages to build up a Modflow model
        
        Parameters
        ----------
        bron_data : dict
            dictionary containing data to build up the model        
        """
        # prepare the modflow modules
        oc  = flopy.modflow.ModflowOc(self.mf)
        bas = flopy.modflow.ModflowBas(
            self.mf, ibound=self.bas["ibound"], strt=self.mf.dis.top.array # self.bas['strt']#
        )
        lpf = flopy.modflow.ModflowLpf(
            self.mf, laytyp=self.lpf["laytyp"], hk=self.lpf["hk"], ipakcb=53
        )  # lPF OR UPW!
        # upw = flopy.modflow.ModflowUpw(mf, laytyp=laytyp, hk=Kx, ss=1e-05, sy=0.15)
        wel = flopy.modflow.ModflowWel(
            self.mf, stress_period_data=self.wel["stress_period_data"]
        )
        pcg = flopy.modflow.ModflowPcg(self.mf)
#         lmt = flopy.modflow.ModflowLmt( 
#             self.mf, output_file_name=f'mt3d_link.ftl'
#         )  # link to MT3DMS
