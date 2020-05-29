# (c) and source https://rdrr.io/github/Issoufou-Liman/decisionSupportExtra/


library(reshape2)

#' Format \code{\link[fitdistrplus]{descdist}} data to ggplot2
#'
#' Generate a list of \code{\link[fitdistrplus]{descdist}} outputs to be understood by ggplot2.
#'
#' @author Issoufou Liman
#' @inheritParams fitdistrplus::descdist
#' @param title,subtitle Title and Subtitle
#' @seealso \code{\link[fitdistrplus]{descdist}}.
#' @details see \code{\link[fitdistrplus]{descdist}}.
#' @references
#' Marie Laure Delignette-Muller, Christophe Dutang (2015). fitdistrplus: An R Package for Fitting Distributions. Journal of Statistical Software, 64(4), 1-34. http://www.jstatsoft.org/v64/i04/.
#' @examples
#' ggbuild_descdist_data(rbeta(100, shape1 = 0.05, shape2 = 1), boot = 500, obs.col = 'blue',
#' boot.col = 'yellow')
#' @importFrom stats median sd
#' @export ggbuild_descdist_data
ggbuild_descdist_data <- function(data, discrete = FALSE, boot = NULL, method = "unbiased", graph = TRUE, 
    title = "Cullen and Frey graph", subtitle = NULL, obs.col = "darkblue", boot.col = "orange") {
    if (missing(data) || !is.vector(data, mode = "numeric")) 
        stop("data must be a numeric vector")
    if (length(data) < 4) 
        stop("data must be a numeric vector containing at least four values")
    moment <- function(data, k) {
        m1 <- mean(data)
        return(sum((data - m1)^k)/length(data))
    }
    if (method == "unbiased") {
        skewness <- function(data) {
            # unbiased estimation (Fisher 1930)
            sd <- sqrt(moment(data, 2))
            n <- length(data)
            gamma1 <- moment(data, 3)/sd^3
            unbiased.skewness <- sqrt(n * (n - 1)) * gamma1/(n - 2)
            return(unbiased.skewness)
        }
        kurtosis <- function(data) {
            # unbiased estimation (Fisher 1930)
            n <- length(data)
            var <- moment(data, 2)
            gamma2 <- moment(data, 4)/var^2
            unbiased.kurtosis <- (n - 1)/((n - 2) * (n - 3)) * ((n + 1) * gamma2 - 3 * (n - 1)) + 3
            return(unbiased.kurtosis)
        }
        standdev <- function(data) {
            sd(data)
        }
    } else if (method == "sample") {
        skewness <- function(data) {
            sd <- sqrt(moment(data, 2))
            return(moment(data, 3)/sd^3)
        }
        kurtosis <- function(data) {
            var <- moment(data, 2)
            return(moment(data, 4)/var^2)
        }
        standdev <- function(data) {
            sqrt(moment(data, 2))
        }
    } else stop("The only possible value for the argument method are 'unbiased' or 'sample'")
    
    res <- list(min = min(data), max = max(data), median = median(data), mean = mean(data), sd = standdev(data), 
        skewness = skewness(data), kurtosis = kurtosis(data), method = method)
    
    
    skewdata <- res$skewness
    kurtdata <- res$kurtosis
    dist_graph <- function() {
        # Cullen and Frey graph
        if (graph) {
            # bootstrap sample for observed distribution and computation of kurtmax from this sample
            if (!is.null(boot)) {
                if (!is.numeric(boot) || boot < 10) {
                  stop("boot must be NULL or a integer above 10")
                }
                n <- length(data)
                
                databoot <- matrix(sample(data, size = n * boot, replace = TRUE), nrow = n, ncol = boot)
                s2boot <- sapply(1:boot, function(iter) skewness(databoot[, iter])^2)
                kurtboot <- sapply(1:boot, function(iter) kurtosis(databoot[, iter]))
                
                kurtmax <- max(10, ceiling(max(kurtboot)))
                xmax <- max(4, ceiling(max(s2boot)))
            } else {
                kurtmax <- max(10, ceiling(kurtdata))
                xmax <- max(4, ceiling(skewdata^2))
            }
            
            ymax <- kurtmax - 1
            yax <- as.character(kurtmax - 0:ymax)
            
            par_skew_kurto_data = list(xmax = xmax, ymax = ymax, yax = yax)
            
            skew_kurto_data <- data.frame(x = (skewdata^2), y = (kurtmax - kurtdata))
            data_list <- list(skew_kurto_data = skew_kurto_data)
            if (!discrete) {
                # beta dist
                p <- exp(-100)
                lq <- seq(-100, 100, 0.1)
                q <- exp(lq)
                s2a <- (4 * (q - p)^2 * (p + q + 1))/((p + q + 2)^2 * p * q)
                ya <- kurtmax - (3 * (p + q + 1) * (p * q * (p + q - 6) + 2 * (p + q)^2)/(p * q * (p + 
                  q + 2) * (p + q + 3)))
                p <- exp(100)
                lq <- seq(-100, 100, 0.1)
                q <- exp(lq)
                s2b <- (4 * (q - p)^2 * (p + q + 1))/((p + q + 2)^2 * p * q)
                yb <- kurtmax - (3 * (p + q + 1) * (p * q * (p + q - 6) + 2 * (p + q)^2)/(p * q * (p + 
                  q + 2) * (p + q + 3)))
                s2 <- c(s2a, s2b)
                y <- c(ya, yb)
                beta_dist <- data.frame(x = s2, y = y)
                data_list$beta_dist <- beta_dist
                # gamma dist
                lshape <- seq(-100, 100, 0.1)
                shape <- exp(lshape)
                s2 <- 4/shape
                y <- kurtmax - (3 + 6/shape)
                gamma_dist <- data.frame(x = s2, y = y)
                data_list$gamma_dist <- gamma_dist
                # log-normal dist
                lshape <- seq(-100, 100, 0.1)
                shape <- exp(lshape)
                es2 <- exp(shape^2)
                s2 <- (es2 + 2)^2 * (es2 - 1)
                y <- kurtmax - (es2^4 + 2 * es2^3 + 3 * es2^2 - 3)
                lnorm_dist <- data.frame(x = s2, y = y)
                data_list$lnorm_dist <- lnorm_dist
            } else {
                # negative binomial dist
                p <- exp(-10)
                lr <- seq(-100, 100, 0.1)
                r <- exp(lr)
                s2a <- (2 - p)^2/(r * (1 - p))
                ya <- kurtmax - (3 + 6/r + p^2/(r * (1 - p)))
                p <- 1 - exp(-10)
                lr <- seq(100, -100, -0.1)
                r <- exp(lr)
                s2b <- (2 - p)^2/(r * (1 - p))
                yb <- kurtmax - (3 + 6/r + p^2/(r * (1 - p)))
                s2 <- c(s2a, s2b)
                y <- c(ya, yb)
                negbin_dist <- data.frame(x = s2, y = y)
                data_list$negbin_dist <- negbin_dist
                
                # poisson dist
                llambda <- seq(-100, 100, 0.1)
                lambda <- exp(llambda)
                s2 <- 1/lambda
                y <- kurtmax - (3 + 1/lambda)
                poisson_dist <- data.frame(x = s2, y = y)
                data_list$poisson_dist <- poisson_dist
            }
            # bootstrap sample for observed distribution
            if (!is.null(boot)) {
                boot_data <- data.frame(x = s2boot, y = kurtmax - kurtboot)
                data_list$boot_data <- boot_data
            }
            # observed dist
            observed_dist <- data.frame(x = skewness(data)^2, y = kurtmax - kurtosis(data))
            data_list$observed_dist <- observed_dist
            # normal dist
            norm_dist <- data.frame(x = 0, y = kurtmax - 3)
            data_list$norm_dist <- norm_dist
            # uniform dist
            unif_dist <- data.frame(x = 0, y = kurtmax - 9/5)
            data_list$unif_dist <- unif_dist
            # exponential dist
            exp_dist <- data.frame(x = 2^2, y = kurtmax - 9)
            data_list$exp_dist <- exp_dist
            # logistic dist
            logistic_dist <- data.frame(x = 0, y = kurtmax - 4.2)
            data_list$logistic_dist <- logistic_dist
        }
        data_list <- list(data_list = data_list, par_skew_kurto_data = par_skew_kurto_data)
    }
    data_list <- dist_graph()
    data_list
}

#' ggplot Empirical distribution as in \code{\link[fitdistrplus]{descdist}}
#'
#' Generate compact plot following the \code{\link[fitdistrplus]{descdist}} function as customizable ggplot.
#'
#' @author Issoufou Liman
#' @inheritParams fitdistrplus::descdist
#' @param title,subtitle Title and Subtitle
#' @param xlab,ylab These are respectively x and y labels.
#' @param obs_geom_size,boot_geom_size,dist_geom_pts_size The size of the geom_point to be used for the empirical distributoion (default to 4), bootstrapping (default to 0.02), theoritical distribution (default to 5), respectively.
#' @param dist_geom_line_size The size of the geom_line to be used for the empirical distributoion. The default is 0.6.
#' @param axis_text_size,axis_title_size,plot_title_size,plot_subtitle_size,strip_text_size,legend_text_size = 12,
#' Text size respectively corresponding to axis text (default to 12), axis title (default to 12), plot title (default to 20), subtitle (default to 17), strip text (default to 18), and legend (default to 12).
#' @seealso \code{\link[fitdistrplus]{descdist}}.
#' @details see \code{\link[fitdistrplus]{descdist}}.
#' @references
#' Marie Laure Delignette-Muller, Christophe Dutang (2015). fitdistrplus: An R Package for Fitting Distributions. Journal of Statistical Software, 64(4), 1-34. http://www.jstatsoft.org/v64/i04/.
#' @examples
#' ggplot_descdist(rbeta(100, shape1 = 0.05, shape2 = 1), boot = 500, obs.col = "blue",
#' boot.col = "yellow")
#' @importFrom scales rescale_none
#' @importFrom reshape2 melt
#' @export ggplot_descdist
ggplot_descdist <- function(data, boot = 1000, obs.col = "darkblue", boot.col = "orange",
                            title = "Cullen and Frey graph", subtitle = NULL,
                            xlab = "square of skewness", ylab = "kurtosis",
                            obs_geom_size = 4, boot_geom_size = 0.02, dist_geom_pts_size = 5,
                            dist_geom_line_size = 0.6, axis_text_size = 12, axis_title_size = 12,
                            plot_title_size = 20, plot_subtitle_size = 17,strip_text_size = 18,
                            legend_text_size = 12){

  # data <- ifelse(is.data.frame(data), data, as.data.frame(data)) # not sure why it is not working
  if(!is.data.frame(data)) data <- as.data.frame(data)
  plot_lims <- sapply(data, function(i){
    ggbuild_descdist_data(i, boot = boot, graph = TRUE)
  }, simplify = FALSE, USE.NAMES = TRUE)

  fit_data <- lapply(plot_lims, '[[', 1)
  plot_lims <- lapply(plot_lims, '[[', 2)
  yax <- unique(unlist(lapply(plot_lims, '[[', 'yax')))
  ymax <- max(unlist(lapply(plot_lims, '[[', 'ymax')))
  xmax <- max(unlist(lapply(plot_lims, '[[', 'xmax')))


  # names_plot_lims <- as.character(sapply(plot_lims, names))
  # plot_lims <- unlist(plot_lims, use.names = TRUE)
  # names(plot_lims) <- names_plot_lims
  # plot_lims

  fit_data <- melt(fit_data, id.var=c("x", "y"))
  fit_data$group <- sapply(fit_data$L2, function(i){
    check <- unique(fit_data$L2)
    which(check == i)
  })

  my_theme <- theme_minimal() +
    theme(
      text = element_text(family = 'sans', size = 16, face = 'plain'),
      #panel.grid.major = element_blank(),
      #panel.grid.minor = element_blank(),
      panel.spacing=unit(0.075, "lines"),
      panel.border = element_rect(color = "lightgrey", fill = NA, size = 0.75),
      axis.ticks = element_line(colour = 'black', size = 0.075),
      axis.text = element_text(size = axis_text_size),
      axis.title = element_text(size = axis_title_size),
      # legend.title = element_blank(),
      legend.text = element_text(size = legend_text_size),
      legend.position="top",
      legend.justification = 'right',
      legend.margin=margin(0, 0, 0, 0),
      legend.box.margin=margin(-22, 0, -10, 0),
      strip.text = element_text(size = strip_text_size),
      # strip.background = element_rect(color = "gray", size = 0.075),
      strip.background = element_rect(size = 0.075, fill='lightgoldenrodyellow'),

      plot.title = element_text(size=plot_title_size, color = 'grey'),
      plot.subtitle = element_text(size=plot_subtitle_size, face="italic", color="yellow3"),
      plot.background = element_rect(size=0.13,linetype="solid", color="black")
    )
  # fit_data$L1 <- factor(fit_data$L1, levels = unique(fit_data$L1))
  ggplot() +
    geom_polygon(data = fit_data[fit_data$L2 == 'beta_dist', ],
                 aes_string("x", "y", group = "L1", alpha = factor("beta"))) +
    geom_line(data = fit_data[fit_data$L2 == 'lnorm_dist', ],
              aes_string("x", "y", linetype = shQuote('lognormal')), size = dist_geom_line_size)+
    geom_line(data = fit_data[fit_data$L2 == 'gamma_dist', ],
              aes_string("x", "y", linetype = shQuote('gamma')), size = dist_geom_line_size)+

    geom_point(data = fit_data[fit_data$L2 == 'boot_data', ],
               aes_string("x", "y", color = shQuote("Bootstrapped values")), size = boot_geom_size)+
    geom_point(data = fit_data[fit_data$L2 == 'observed_dist', ],
               aes_string("x", "y", color = shQuote("Observation")), size = obs_geom_size)+

    # geom_point(data = fit_data[fit_data$L2 == 'skew_kurto_data', ],
    #            aes(x, y, shape = 'a'), colour = 'black', size = 5)+

    geom_point(data = fit_data[fit_data$L2 == 'norm_dist', ],
               aes_string("x", "y", shape = shQuote("normal")), size = dist_geom_pts_size)+
    geom_point(data = fit_data[fit_data$L2 == 'unif_dist', ],
               aes_string("x", "y", shape = shQuote("uniform")), size = dist_geom_pts_size)+
    geom_point(data = fit_data[fit_data$L2 == 'exp_dist', ],
               aes_string("x", "y", shape = shQuote("exponential")), size = dist_geom_pts_size)+
    geom_point(data = fit_data[fit_data$L2 == 'logistic_dist', ],
               aes_string("x", "y", shape = shQuote("logistic")), size = dist_geom_pts_size)+


    scale_colour_manual(values=c(boot.col, obs.col)) +
    scale_linetype_manual(values=c('dashed', 'dotted'))+
    scale_alpha_manual(values = 0.3) +
    scale_shape_manual(values = c(8, 2, 7, 3))+
    # scale_size_identity()+ # this would remove cause the legend to not show if size was supplied as aes

    guides(color = guide_legend(title = NULL, order = 1, ncol = 1,
                                override.aes = list(color = c(boot.col, obs.col))),
           alpha = guide_legend(title = NULL, order = 3),
           shape = guide_legend(title = NULL, order = 2, ncol = 2,
                                override.aes = list(shape = c(8, 2, 7, 3))),
           linetype = guide_legend(title = "(Weibull is close to gamma and lognormal)",
                                   title.position = "bottom",
                                   title.theme = element_text(size = legend_text_size-1),
                                   order = 4, ncol = 1,
                                   override.aes = list(linetype = c('dashed', 'dotted'))))+

    facet_wrap(.~L1)+
    labs(title = title, subtitle = subtitle,
         x = xlab,
         y = ylab)+
    my_theme +
    scale_x_continuous(breaks = seq(0, ymax, by=ymax/10), limits = c(0,xmax), oob = rescale_none)+
    scale_y_continuous(breaks = seq(min(as.numeric(yax)), max(as.numeric(yax)), by=(max(as.numeric(yax)) - min(as.numeric(yax))) / 10), limits = c(0, ymax), oob = rescale_none)
}