package miniJava.interpreter.miniJava.impl;

import java.lang.Override;
import miniJava.interpreter.miniJava.MiniJavaPackage;
import miniJava.interpreter.miniJava.Parameter;
import miniJava.interpreter.miniJava.TypeRef;
import org.eclipse.emf.ecore.EClass;
import org.eclipse.emf.ecoretools.ale.compiler.lib.EqualService;

public class ParameterImpl extends SymbolImpl implements Parameter {
	protected ParameterImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.PARAMETER;
	}

	public boolean compare(Parameter other) {
		boolean result;
		result = (boolean) (((EqualService.equals((this.name), (other.getName()))) && (((TypeRef) (this.typeRef)).compare((TypeRef) (other.getTypeRef()))))) ;
		return result;
	}
}
