# Copied colab cells in case it dies

class Source():
    '''Astronomical source object for NOAO formatted light curve'''
    
    def __init__(self, filename):
        '''Read in light curve data
        PARAMETERS: filename -- string giving file location
        '''
        lc_df = pd.read_csv(filename, delim_whitespace=True, comment = '#')
        u_obs = np.where(lc_df['pb'] == 'u')
        if len(u_obs[0]) > 0:
            lc_df.drop(lc_df.index[u_obs], inplace=True)
        self._lc_df = lc_df
        self._filename = filename
        self._filters = np.unique(self._lc_df['pb'])
        
    def plot_multicolor_lc(self):
        '''Plot the 4 band light curve'''
        fig, ax = plt.subplots(figsize=(7,7))
        color_dict = {'g': '#78A5A3', 
                      'r': '#CE5A57', 
                      'i': '#E1B16A', 
                      'z': '#444C5C', }
        
        for filt in self._filters:
          lc_filt = self._lc_df[self._lc_df['pb'] == filt]
          ax.errorbar(lc_filt['t'], lc_filt['flux'], lc_filt['dflux'], fmt = 'o', color=color_dict[filt], label=filt)
            
        ax.legend(fancybox = True)
        ax.set_xlabel(r"$\mathrm{MJD}$")
        ax.set_ylabel(r"$\mathrm{flux}$")
        fig.tight_layout()


class Variable(Source):  #inherit properties of Source class
    '''Variable subclass of fake sources'''
    
    def __init__(self, filename):
        """Initialize attributes of the variable object"""
        Source.__init__(self, filename)
        self.Std = self.normalized_flux_std()
        self.Amp = self.normalized_amplitude()
        self.MAD = self.normalized_MAD()
        self.beyond1std = self.beyond_1std()
        self.skew = self.skew()
        colors = self.mean_colors()
        
        
    def filter_flux(self):
        '''Store individual passband fluxes as object attributes'''
        
        for filt in self._filters:
            filt_obs = np.where(self._lc_df['pb'] == filt)
            exec("self._{0}_time = self._lc_df['t'].iloc[filt_obs]".format(filt))
            exec("self._{0}_flux = self._lc_df['flux'].iloc[filt_obs]".format(filt))
            exec("self._{0}_flux_unc = self._lc_df['dflux'].iloc[filt_obs]".format(filt))

    def weighted_mean_flux(self):
        '''Measure (SNR weighted) mean flux in griz'''
        if not hasattr(self, '_g_flux'):
            self.filter_flux()
            
        weighted_mean = lambda flux, dflux: np.sum(flux*(flux/dflux)**2)/np.sum((flux/dflux)**2)
        
        for filt in self._filters:
            exec("self._{0}_mean = weighted_mean(self._{0}_flux, self._{0}_flux_unc)".format(filt))
        
    def normalized_flux_std(self):
        '''Collect flux and normalize by weighted fean flux for each filt'''
        if not hasattr(self, '_g_mean'):
            self.weighted_mean_flux()
        
        get_std = lambda flux, mean: np.std(flux) / mean

        for filt in self._filters: 
            exec('self._{0}_std = get_std(self._{0}_flux, self._{0}_mean)'.format(filt))
        

    def normalized_amplitude(self):
        '''Collect the maximum absolute value flux and normaize'''
        if not hasattr(self, '_g_mean'):
            self.weighted_mean_flux()
            
        get_amp = lambda flux, mean: np.max(np.absolute(flux)) / mean
        
        for filt in self._filters:
            exec('self._{0}_amp = get_amp(self._{0}_flux, self._{0}_mean)'.format(filt))
        
        
    def normalized_MAD(self):
        '''Collect the median absolute value flux and normalize'''
        
        if not hasattr(self, '_g_mean'):
            self.weighted_mean_flux()
            
        get_amp = lambda flux, mean: np.median(np.absolute(flux)) / mean
        
        for filt in self._filters:
            exec('self._{0}_amp = get_amp(self._{0}_flux, self._{0}_mean)'.format(filt))
        
        
    def beyond_1std(self):
        '''Collect fraction of flux measurements with flux outside mean +/- 1 std'''
        if not hasattr(self, '_g_std'):
            self.normalized_flux_std()
            
        #correct for the normailzed std when making the comparison
        get_frac = lambda flux, mean, std: flux[(flux > mean + std * mean) | (flux < mean - std * mean)].shape[0] / flux.shape[0]
        
        for filt in self._filters:
            exec('self._{0}_beyond1std = get_frac(self._{0}_flux, self._{0}_mean, self._{0}_std)'.format(filt))
    
    def skew(self):
        '''Calcultate the skew of the flux measurements in each filter'''
        if not hasattr(self, '_g_flux'):
            self.filter_flux()
            
        get_skew = lambda flux: spstat.skew(flux)
        
        for filt in self._filters:
            exec("self._{0}_skew = get_skew(self._{0}_flux)".format(filt))
            
    def mean_colors(self):
        '''Collect mean g-r, r-i, and i-z colors, units of flux'''
        if not hasattr(self, '_g_mean'):
            self.weighted_mean_flux()
            
        color_pairs = [('g', 'r'), ('r', 'i'), ('i', 'z')]
        
        get_color = lambda flux1, flux2: flux1 - flux2
        
        for pair in color_pairs:
            exec("self._%s_%s_mean = get_color(self._%s_mean, self._%s_mean)" %(pair[0], pair[1], pair[0], pair[1]))
        
