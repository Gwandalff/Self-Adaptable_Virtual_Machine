package fr.gjouneau.selfadaptable_minijava.lang.interpreter.execute;

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

public class App 
{
	
	static public String program;
	static private Program minijavaProgram;
	
    public static void main( String[] args )
    {
    	program = args[0];
    	if(program.endsWith(".xmi")) {
    		loadXMI();
    	} else {
//    		loadMiniJava();
    	}
    	
    	minijavaProgram.execute();
    }
    
    public static void loadXMI() {
		EPackage.Registry.INSTANCE.put("http://miniJava.miniJava.miniJava/", MiniJavaPackage.eINSTANCE);
		final Resource.Factory.Registry reg = Resource.Factory.Registry.INSTANCE;
		final Map<String, Object> m = reg.getExtensionToFactoryMap();
		m.put("xmi", new XMIResourceFactoryImpl());

		LogService.MUTE = false;

		final ResourceSetImpl resSet = new ResourceSetImpl();
		final URI createFileURI = URI.createFileURI(program);
		final Resource resource = resSet.getResource(createFileURI, true);
		
		FeedbackLoop.registerModule(new ApproximateModule());

		minijavaProgram = (Program) resource.getContents().get(0);
		minijavaProgram.initialize(new BasicEList());
	}
    
//    public static void loadMiniJava() {
//    	Injector injector = new MiniJavaStandaloneSetup().createInjectorAndDoEMFRegistration();
//    	XtextResourceSet resourceSet = injector.getInstance(XtextResourceSet.class);
//    	resourceSet.addLoadOption(XtextResource.OPTION_RESOLVE_ALL, Boolean.TRUE);
//    	
//    	final URI createFileURI = URI.createFileURI(program);
//		final Resource resource = resourceSet.getResource(createFileURI, true);
//		
//    	minijavaProgram = (Program) resource.getContents().get(0);
//    	minijavaProgram.initialize(new BasicEList());
//    }
}
