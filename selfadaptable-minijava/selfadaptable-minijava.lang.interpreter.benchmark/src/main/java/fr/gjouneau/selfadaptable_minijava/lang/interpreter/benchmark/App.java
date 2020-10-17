package fr.gjouneau.selfadaptable_minijava.lang.interpreter.benchmark;

import java.util.Map;

import org.eclipse.emf.common.util.BasicEList;
import org.eclipse.emf.common.util.URI;
import org.eclipse.emf.ecore.EPackage;
import org.eclipse.emf.ecore.resource.Resource;
import org.eclipse.emf.ecore.resource.impl.ResourceSetImpl;
import org.eclipse.emf.ecore.xmi.impl.XMIResourceFactoryImpl;
import org.eclipse.emf.ecoretools.ale.compiler.lib.LogService;

import miniJava.interpreter.FeedbackLoop;
import miniJava.interpreter.dynamicModules.ApproximateModule;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Program;

import org.openjdk.jmh.annotations.*;


@State(Scope.Benchmark)
public class App 
{
	@Param({"/home/benchmarks/programs/minijava_sobel.xmi"})
	//@Param({"../programs/sobel.xmi"})
	public String program;
	private Program minijavaProgram;
    
    @Setup(Level.Iteration)
    public void loadXMI() {
		EPackage.Registry.INSTANCE.put("http://miniJava.miniJava.miniJava/", MiniJavaPackage.eINSTANCE);
		final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
		final Map<String, Object> m = reg.getExtensionToFactoryMap();
		m.put("xmi", new XMIResourceFactoryImpl());

		LogService.MUTE = true;

		final ResourceSetImpl resSet = new ResourceSetImpl();
		final URI createFileURI = URI.createFileURI(program);
		final Resource resource = resSet.getResource(createFileURI, true);
		
		FeedbackLoop.reset();
		FeedbackLoop.registerModule(new ApproximateModule());

		minijavaProgram = (Program) resource.getContents().get(0);
		minijavaProgram.initialize(new BasicEList());
	}
    
	@Benchmark
    @BenchmarkMode(Mode.AverageTime)
    @Measurement(iterations = 30)
    @Fork(value = 1, jvmArgs = {"-Xms4G", "-Xmx4G"})
    @Warmup(iterations = 10)
    public miniJava.interpreter.miniJava.State minijavaInterpreter() {

        return minijavaProgram.execute();
    }
}
