package io.springbok.orekit_demos;

import java.io.BufferedReader;
import java.io.File;
import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URISyntaxException;
import java.nio.file.Paths;
import java.util.ArrayList;

import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.linear.DiagonalMatrix;
import org.hipparchus.ode.nonstiff.AdaptiveStepsizeIntegrator;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.hipparchus.optim.nonlinear.vector.leastsquares.GaussNewtonOptimizer;
import org.hipparchus.random.CorrelatedRandomVectorGenerator;
import org.hipparchus.random.GaussianRandomGenerator;
import org.hipparchus.random.ISAACRandom;
import org.hipparchus.util.FastMath;
import org.orekit.bodies.BodyShape;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.bodies.OneAxisEllipsoid;
import org.orekit.data.DataContext;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.estimation.iod.IodLambert;
import org.orekit.estimation.leastsquares.BatchLSEstimator;
import org.orekit.estimation.measurements.AngularAzEl;
import org.orekit.estimation.measurements.GroundStation;
import org.orekit.estimation.measurements.ObservableSatellite;
import org.orekit.estimation.measurements.Range;
import org.orekit.estimation.measurements.generation.AngularAzElBuilder;
import org.orekit.estimation.measurements.generation.RangeBuilder;
import org.orekit.forces.ForceModel;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.frames.TopocentricFrame;
import org.orekit.frames.Transform;
import org.orekit.orbits.KeplerianOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngle;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.analytical.KeplerianPropagator;
import org.orekit.propagation.analytical.tle.TLE;
import org.orekit.propagation.analytical.tle.TLEPropagator;
import org.orekit.propagation.conversion.DormandPrince853IntegratorBuilder;
import org.orekit.propagation.conversion.NumericalPropagatorBuilder;
import org.orekit.propagation.integration.AbstractIntegratedPropagator;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.TimeStampedPVCoordinates;

/**
 * Demonstrate orbit propagation, measurement, and estimation using Orekit.
 *
 */
public class App {

	/**
	 * Compute position in a specified frame given a measurement in a station frame.
	 * 
	 * @param range range
	 * @param azel  azimuth and elevation
	 * @param staF  station frame
	 * @param posF  position frame
	 * @return inertial position
	 */
	public static Vector3D getPositionFromRangeAzEl(Range range, AngularAzEl azel, Frame staF, Frame posF) {
		if (range.getDate().compareTo(azel.getDate()) != 0) {
			throw new IllegalArgumentException("Measurements dates are not equal");
		}
		double ra = range.getObservedValue()[0];
		double az = azel.getObservedValue()[0]; // Clockwise from North, viewed from above
		double el = azel.getObservedValue()[1];
		double alpha = Math.PI / 2.0 - az; // Counter-clockwise from East, viewed from above
		double delta = el;
		Vector3D rHat = new Vector3D(alpha, delta); // Unit vector from station to satellite
		Vector3D pos = new Vector3D(ra, rHat); // Position vector from station to satellite
		Transform topographicToInertial = staF.getTransformTo(posF, range.getDate());
		Vector3D inertialPos = topographicToInertial.transformPosition(pos);
		return inertialPos;
	}

	/**
	 * Convert state to a string reporting range, azimuth, and elevation in the
	 * specified station frame.
	 * 
	 * @param state spacecraft state
	 * @param staF  station frame
	 * @return range, azimuth, and elevation string
	 */
	public static String stateToRangeAzElStr(SpacecraftState state, Frame staF) {
		TimeStampedPVCoordinates pv = state.getPVCoordinates(staF);
		Vector3D pos = pv.getPosition();
		double ra = pos.getNorm();
		double az = Math.PI / 2.0 - pos.getAlpha();
		double el = pos.getDelta();
		return String.format("%12.1f %12.6f %12.6f", ra, az, el);
	}

	/**
	 * Read a TLE, construct a Keplerian orbit and ground station, propagate orbit
	 * using a variety of propagators and build measurements, then perform initial
	 * orbit determination and least squares estimation.
	 * 
	 * @param args
	 */
	public static void main(String[] args) {

		// Propagation duration, seconds
		double duration = 600.0;

		// Propagation step, seconds
		double stepT = 60.0;

		// Selected gravitation coefficient
		double mu = Constants.IERS2010_EARTH_MU;

		// Manage Orekit data context
		File orekitData = null;
		try {
			orekitData = Paths.get(App.class.getResource("/orekit-data").toURI()).toFile();
		} catch (URISyntaxException e) {
			System.out.println("Could not get Orekit data: " + e.getMessage());
			System.exit(1);
		}
		DataProvidersManager manager = DataContext.getDefault().getDataProvidersManager();
		manager.addProvider(new DirectoryCrawler(orekitData));

		// Read TLE file
		InputStream stream = App.class.getResourceAsStream("/tles-43249-as-of-2020-05-04-100834.txt");
		BufferedReader reader = new BufferedReader(new InputStreamReader(stream));
		String line1 = null;
		String line2 = null;
		TLE tle = null;
		try {
			line1 = reader.readLine();
			line2 = reader.readLine();
			reader.close();
		} catch (IOException e) {
			System.out.println("Could not read TLE file: " + e.getMessage());
			System.exit(1);
		}
		tle = new TLE(line1, line2);

		// Construct a Keplerian orbit
		double a = Math.cbrt(mu / (Math.pow(tle.getMeanMotion(), 2)));
		double e = tle.getE();
		double i = tle.getI();
		double omega = tle.getPerigeeArgument();
		double raan = tle.getRaan();
		double lM = tle.getMeanAnomaly();
		AbsoluteDate initialDate = tle.getDate(); // UTC
		Frame inertialF = FramesFactory.getGCRF();
		Orbit initialOrbit = new KeplerianOrbit(a, e, i, omega, raan, lM, PositionAngle.MEAN, inertialF, initialDate,
				mu);
		SpacecraftState initialState = new SpacecraftState(initialOrbit);

		// Construct a ground station
		double longitude = FastMath.toRadians(45.);
		double latitude = FastMath.toRadians(25.);
		double altitude = 0.;
		GeodeticPoint staP = new GeodeticPoint(latitude, longitude, altitude);
		Frame earthF = FramesFactory.getITRF(IERSConventions.IERS_2010, true);
		BodyShape earth = new OneAxisEllipsoid(Constants.WGS84_EARTH_EQUATORIAL_RADIUS,
				Constants.WGS84_EARTH_FLATTENING, earthF);
		TopocentricFrame staF = new TopocentricFrame(earth, staP, "station");
		GroundStation groundStation = new GroundStation(staF);
		groundStation.getPrimeMeridianOffsetDriver().setReferenceDate(initialDate);
		groundStation.getPolarOffsetXDriver().setReferenceDate(initialDate);
		groundStation.getPolarOffsetYDriver().setReferenceDate(initialDate);

		// Set up ground station measurements builders (for range, azimuth, and
		// elevation)
		int small = 0;
		ISAACRandom randomNumGenerator = new ISAACRandom();
		GaussianRandomGenerator gaussianGenerator = new GaussianRandomGenerator(randomNumGenerator);
		int satIdx = 0;
		ObservableSatellite obsSat = new ObservableSatellite(satIdx);
		AbsoluteDate finalDate = initialDate.shiftedBy(duration);

		double eps = 1.0e-6;

		double[] rangeMatrix = new double[] { eps, eps };
		DiagonalMatrix rangeCovarianceMatrix = new DiagonalMatrix(rangeMatrix);
		CorrelatedRandomVectorGenerator rangeNoiseGenerator = new CorrelatedRandomVectorGenerator(rangeCovarianceMatrix,
				small, gaussianGenerator);
		double sigma = 1.0;
		double baseWeight = 1.0;
		RangeBuilder rangeBuilder = new RangeBuilder(rangeNoiseGenerator, groundStation, false, sigma, baseWeight,
				obsSat);
		rangeBuilder.init(initialDate, finalDate);
		ArrayList<Range> rangeContainer = new ArrayList<Range>();

		double[] angularMatrix = new double[] { eps, eps };
		DiagonalMatrix angularCovarianceMatrix = new DiagonalMatrix(angularMatrix);
		CorrelatedRandomVectorGenerator angularNoiseGenerator = new CorrelatedRandomVectorGenerator(
				angularCovarianceMatrix, small, gaussianGenerator);
		double[] azElSigmas = new double[] { 1.0, 1.0 };
		double[] azElBaseWeights = new double[] { 1.0, 1.0 };
		AngularAzElBuilder azElBuilder = new AngularAzElBuilder(angularNoiseGenerator, groundStation, azElSigmas,
				azElBaseWeights, obsSat);
		azElBuilder.init(initialDate, finalDate);
		ArrayList<AngularAzEl> azElContainer = new ArrayList<AngularAzEl>();

		// Construct a Keplerian propagator
		KeplerianPropagator kPropagator = new KeplerianPropagator(initialOrbit);

		// Construct a TLE propagator
		TLEPropagator sPropagator = TLEPropagator.selectExtrapolator(tle);

		// Construct a numerical propagator
		double minStep = 0.001;
		double maxStep = 1000.0;
		double positionTolerance = 10.0;
		OrbitType propagationType = OrbitType.KEPLERIAN;
		double[][] tolerances = NumericalPropagator.tolerances(positionTolerance, initialOrbit, propagationType);
		AdaptiveStepsizeIntegrator integrator = new DormandPrince853Integrator(minStep, maxStep, tolerances[0],
				tolerances[1]);
		NumericalPropagator nPropagator = new NumericalPropagator(integrator);
		nPropagator.setOrbitType(propagationType);
		NormalizedSphericalHarmonicsProvider provider = GravityFieldFactory.getNormalizedProvider(10, 10);
		ForceModel holmesFeatherstone = new HolmesFeatherstoneAttractionModel(
				FramesFactory.getITRF(IERSConventions.IERS_2010, true), provider);
		nPropagator.addForceModel(holmesFeatherstone);
		nPropagator.setInitialState(initialState);

		// Propagate orbit
		int iStep = 0;
		Vector3D initialPosition = null;
		Vector3D finalPosition = null;
		for (AbsoluteDate extrapDate = initialDate; extrapDate.compareTo(finalDate) <= 0; extrapDate = extrapDate
				.shiftedBy(stepT)) {
			iStep++;
			SpacecraftState kState = kPropagator.propagate(extrapDate);
			SpacecraftState sState = sPropagator.propagate(extrapDate);
			SpacecraftState nState = nPropagator.propagate(extrapDate);

			// Build measurements (range, azimuth, and elevation)
			SpacecraftState[] states = new SpacecraftState[] { nState };
			Range range = rangeBuilder.build(states);
			AngularAzEl azEl = azElBuilder.build(states);
			rangeContainer.add(range);
			azElContainer.add(azEl);

			System.out.println("step: " + iStep + ", date: " + extrapDate);
			System.out.println("keplerian: " + stateToRangeAzElStr(kState, staF));
			System.out.println("sgp4/sdp4: " + stateToRangeAzElStr(sState, staF));
			System.out.println("numerical: " + stateToRangeAzElStr(nState, staF));
			System.out.println(String.format(" measured: %12.1f %12.6f %12.6f", range.getObservedValue()[0],
					azEl.getObservedValue()[0], azEl.getObservedValue()[1]));
			System.out.println("");

			// Get first and last measurements for IOD
			if (extrapDate.compareTo(initialDate) == 0) {
				initialPosition = getPositionFromRangeAzEl(range, azEl, staF, inertialF);
			} else if (extrapDate.compareTo(finalDate) == 0) {
				finalPosition = getPositionFromRangeAzEl(range, azEl, staF, inertialF);
			}
		}

		// Print initial orbit
		System.out.println("Initial orbit: ");
		System.out.println(initialOrbit.toString());

		// Perform initial orbit determination
		IodLambert lambert = new IodLambert(mu);
		// TODO: Calculate posigrade and number of revolutions
		boolean posigrade = true;
		int nRev = 0;
		KeplerianOrbit orbitEstimation = lambert.estimate(inertialF, posigrade, nRev, initialPosition, initialDate,
				finalPosition, finalDate);
		System.out.println("Lambert IOD estimation: ");
		System.out.println(orbitEstimation.toString());

		// Perform batch least squares estimation to correct orbit
		GaussNewtonOptimizer GNOptimizer = new GaussNewtonOptimizer();
		DormandPrince853IntegratorBuilder dormandPrinceBuilder = new DormandPrince853IntegratorBuilder(minStep, maxStep,
				positionTolerance);
		double positionScale = 1.0;
		final NumericalPropagatorBuilder propBuilder = new NumericalPropagatorBuilder(orbitEstimation,
				dormandPrinceBuilder, PositionAngle.MEAN, positionScale);
		BatchLSEstimator leastSquares = new BatchLSEstimator(GNOptimizer, propBuilder);
		leastSquares.setMaxIterations(1000);
		leastSquares.setMaxEvaluations(1000);
		leastSquares.setParametersConvergenceThreshold(.001);
		azElContainer.forEach(measurement -> leastSquares.addMeasurement(measurement));
		rangeContainer.forEach(measurement -> leastSquares.addMeasurement(measurement));
		AbstractIntegratedPropagator[] lSPropagators = leastSquares.estimate();
		System.out.println("Least squares estimation: ");
		System.out.println(lSPropagators[satIdx].getInitialState());
	}
}
