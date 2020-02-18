package miniJava.visitor.miniJava.impl;

import java.lang.Object;
import java.lang.Override;
import miniJava.visitor.miniJava.MiniJavaPackage;
import miniJava.visitor.miniJava.VariableDeclaration;
import org.eclipse.emf.ecore.EClass;
import visitor.VisitorInterface;

public class VariableDeclarationImpl extends SymbolImpl implements VariableDeclaration {
	protected VariableDeclarationImpl() {
		super();
	}

	@Override
	protected EClass eStaticClass() {
		return MiniJavaPackage.Literals.VARIABLE_DECLARATION;
	}

	@Override
	public Object accept(VisitorInterface visitor) {
		return visitor.visitminiJava__VariableDeclaration(this);
	}
}
